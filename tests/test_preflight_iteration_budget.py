import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from comfyrouter import case_execution as runner
from comfyrouter.execution_contract import ExecutionHooks, execution_context
from comfyrouter.types import RequirementSpec


@pytest.mark.parametrize('accepted',[False,True])
def test_only_proven_unsubmitted_preflight_is_excluded(accepted):
    row={'stages':[{'execution':{'status':'success','prompt_id':'upstream'},'outputs':['upstream.png']},
                   {'execution':{'status':'preflight_failed','error':'estimated mask failed visual review',
                                 'prompt_id':'accepted' if accepted else ''}}]}
    assert runner.runtime_parameter_failure(row)==(None if accepted else 'stage_preflight_failed')
    assert runner.runtime_parameter_failure({'execution':{'status':'success'},'passed':False}) is None
    assert runner.runtime_parameter_failure({'execution':{'status':'executor_error','error':'unknown'}}) is None


def test_migrate_three_preflights_preserves_artifact_and_original_evidence(tmp_path):
    image=tmp_path/'upstream.png';image.write_bytes(b'keep upstream')
    attempts=[{'attempt':n,'passed':False,'stages':[
        {'stage_id':'s1','graph':{'original':True},'execution':{'status':'success' if n==1 else 'reused'},
         'outputs':[str(image)]},
        {'stage_id':'s2','graph':{'repaired_version':n},'execution':{'status':'preflight_failed','error':'bad mask'}}]}
        for n in range(1,4)]
    original=copy.deepcopy(attempts)
    report={'attempts':attempts,'max_attempts':3}
    assert runner.recover_runtime_parameter_failure(report)==3
    assert report['attempts']==[] and report['completed_attempts']==0
    assert report['legacy_parameter_attempts']==original
    assert report['parameter_checkpoint']['attempt']==1
    assert report['parameter_checkpoint']['stages'][1]['graph']=={'repaired_version':3}
    assert report['parameter_checkpoint']['stages'][0]['artifact_sha256']
    assert image.read_bytes()==b'keep upstream'


@pytest.mark.parametrize('composed',[False,True])
@pytest.mark.parametrize('recoverable',[False,True])
def test_preflights_use_separate_budget_then_keep_all_three_normal_rounds(tmp_path,monkeypatch,composed,recoverable):
    monkeypatch.setenv('COMFYROUTER_DISABLE_SELF_EVOLUTION','1')
    monkeypatch.setenv('COMFYROUTER_SKIP_TERMINAL_DIAGNOSIS','1')
    monkeypatch.setattr(runner,'repair_plan_limit',lambda:4)
    monkeypatch.setattr(runner,'normalize_exact_image_geometry',lambda *a,**k:None)
    monkeypatch.setattr(runner,'prepare_single_media_inputs',lambda *a,**k:{})
    monkeypatch.setattr(runner,'repair_mode_for_context',lambda **k:{
        'mode':'parameter','allow_topology_repair':False,'requires_evolution':False})
    counts={'upstream':0,'preflight':0,'generated':0}
    def transaction(executor,graph,destination,**kwargs):
        if graph['1']['inputs']['name']=='upstream':
            counts['upstream']+=1
        elif counts['preflight']<(3 if recoverable else 4):
            counts['preflight']+=1
            return {'graph':graph,'execution':{'status':'preflight_failed','error':'bad mask preview'},
                    'outputs':[],'pre_execution_cleanup':{},'post_execution_cleanup':{},'material_video_sampling':None}
        else:counts['generated']+=1
        destination.mkdir(parents=True,exist_ok=True)
        artifact=destination/'output.png';artifact.write_bytes(b'generated artifact')
        return {'graph':graph,'execution':{'status':'success','prompt_id':str(destination)},'outputs':[artifact],
                'pre_execution_cleanup':{},'post_execution_cleanup':{},'material_video_sampling':None}
    monkeypatch.setattr(runner,'execute_stage',transaction)
    attribution={key:{'likelihood':'unknown','evidence':[],'counterfactual':'inspect actual output'}
        for key in ['text_conditioning','source_image_conditioning','mask_conditioning','negative_conditioning','sampler_conditioning']}
    diagnostic={'earliest_failing_stage_id':'s2','repair_stage_id':'s2','contributing_stage_ids':[],
        'stage_assessments':[{'stage_id':'s1','status':'pass','observations':['upstream correct'],'reason':'pass'},
                             {'stage_id':'s2','status':'fail','observations':['bad mask/output'],'reason':'repair'}],
        'rationale':'repair downstream','evidence':['concrete failure'],'conditioning_attribution':attribution,
        'failure_class':'execution','root_causes':['bad mask/output'],'next_action':'parameter_retry',
        'repair_focus':['downstream'],'needs_search':False,'search_queries':[]}
    agent=Mock()
    agent.diagnose_composition_failure.side_effect=lambda **k:copy.deepcopy(diagnostic)
    agent.decide_repair_search.return_value=copy.deepcopy(diagnostic)
    agent.generate_graph_edit_plan.return_value={'skill_ids':[],'selected_workflow':None,
        'edits':[{'action':'set_param','target':'1','key':'strength','value':2}],
        'requires_composition_replan':False}
    evaluator=Mock(side_effect=[{'comfybench':{'resolved':False}},{'comfybench':{'resolved':False}},
                               {'comfybench':{'resolved':True}}])
    executor=SimpleNamespace(validator=None,static_validate=lambda *a,**k:None)
    spec=RequirementSpec('t2i','Generation',('Generation/text_to_image',))
    def graph(name):return {'1':{'class_type':'Dummy','inputs':{'name':name,'strength':1}}}
    report={'run_id':'test','attempts':[]}
    with execution_context(hooks=ExecutionHooks(evaluator=evaluator)):
        if composed:
            states=[{'typed':SimpleNamespace(stage_id=s,instruction='generate',input_type='TEXT',input_types=('TEXT',),
                       output_type='IMAGE',depends_on=(),modality='t2i'),
                     'spec':spec,'implementation':{'operation':'Generation/text_to_image'},'graph':graph(name),
                     'handoff_inputs':[],'handoff_inputs_by_type':{}}
                    for s,name in [('s1','upstream'),('s2','downstream')]]
            runner.run_composition('001','generate',states,agent,executor,tmp_path,report,lambda:None,3)
        else:
            runner.run_single_case('001','generate','t2i',spec,[{'operation':'Generation/text_to_image'}],
                graph('downstream'),graph('downstream'),None,None,None,[],agent,executor,tmp_path,
                report,lambda:None,3,[])
    assert len(report['parameter_failures'])==(3 if recoverable else 4)
    assert report['completed_attempts']==(3 if recoverable else 0)
    assert counts['generated']==evaluator.call_count==(3 if recoverable else 0)
    if composed:assert counts['upstream']==1
    assert report['status']==('passed' if recoverable else 'parameter_repair_exhausted')
    assert [a['attempt'] for a in report['attempts']]==([1,2,3] if recoverable else [])
