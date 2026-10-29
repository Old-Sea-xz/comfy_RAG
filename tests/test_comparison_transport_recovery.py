import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from comfyrouter.comparison_transport_recovery import recover
from comfyrouter.transport_retry import ready_job_index, retryable_api_failure, write_state
from tools.run_routing_comparison import write


def read(path):
    return json.loads(path.read_text())


def fixture(run, status='routing_failed', error="Server error '502 Bad Gateway'", error_type='AgentTransportError'):
    write(run/'protocol.json',{'transport_retry_policy':{'enabled':True,
        'initial_backoff_seconds':60,'max_backoff_seconds':600,'max_recoveries_per_case':5}})
    route={'status':'routing_failed' if status=='routing_failed' else 'routed',
           'error':error,'error_type':error_type,
           'retrieval_evidence':{'query_count':2,'rounds':[{'round':1,'candidates':[]},
                {'round':2,'query':'missing capability','candidates':[{'workflow_id':'W1','similarity':.9}]}]}}
    write(run/'routing/001_v0/route.json',{'methods':{'semantic':route,'direct':{'status':'routed'}}})
    dest=run/'execution/semantic/001_v0'
    write(dest/'outcome.json',{'status':status,'error':error,'error_type':error_type,'resolved':False})
    return dest


def test_legacy_502_restores_pending_selection_and_keeps_other_method(tmp_path):
    dest=fixture(tmp_path)
    write(tmp_path/'claims/whole_request_routing/semantic_001_v0',{'pid':99999999})
    record=recover(tmp_path,'semantic','001_v0')
    assert record['mode']=='preparation' and record['backoff_seconds']==60
    assert not (dest/'outcome.json').exists()
    assert read(tmp_path/'routing/001_v0/route.json')=={'methods':{'direct':{'status':'routed'}}}
    checkpoint=read(tmp_path/'routing/001_v0/semantic_selection_checkpoint.json')
    assert checkpoint['retrieval_evidence']['query_count']==2
    assert not (tmp_path/'claims/whole_request_routing/semantic_001_v0').exists()
    assert (Path(record['archive'])/'outcome.json').is_file()
    assert ready_job_index(tmp_path,[('semantic','002_v0')],record['next_retry_at']-1) is None
    assert ready_job_index(tmp_path,[('semantic','002_v0')],record['next_retry_at'])==0


@pytest.mark.parametrize('code',[400,401,403,404,422])
def test_bad_requests_are_not_requeued_as_network_outages(tmp_path,code):
    dest=fixture(tmp_path,error=f"Client error '{code} Bad Request'")
    assert recover(tmp_path,'semantic','001_v0') is None
    assert (dest/'outcome.json').is_file()


def test_semantic_refusal_is_terminal(tmp_path):
    fixture(tmp_path,error='selection blocked after 3 retrievals',error_type='AgentOutputError')
    assert recover(tmp_path,'semantic','001_v0') is None


def test_evaluation_recovery_reuses_third_artifact_and_budget(tmp_path):
    dest=fixture(tmp_path,status='evaluation_unavailable')
    output=dest/'raw/case001';output.mkdir(parents=True)
    artifact=output/'final.png';artifact.write_bytes(b'generated once')
    stage={'execution':{'status':'success'},'primary_output':str(artifact)}
    attempts=[{'attempt':i,'passed':False,'stages':[stage],
               'comfybench':{'comfybench':{'resolved':False}}} for i in [1,2]]
    attempts.append({'attempt':3,'passed':None,'stages':[stage],
                     'comfybench':{'status':'evaluator_error','error':"Server error '502 Bad Gateway'"}})
    report={'trace':{'case_output':str(output)},'status':'evaluation_unavailable',
            'max_attempts':3,'attempts':attempts}
    write(output/'run_report.json',report)
    record=recover(tmp_path,'semantic','001_v0')
    saved=read(output/'run_report.json')
    assert record['mode']=='service'
    assert len(saved['attempts'])==2 and saved['max_attempts']==3
    assert saved['service_checkpoint']['attempt']==3
    assert saved['service_checkpoint']['comfyui_attempt_consumed'] is False
    assert artifact.read_bytes()==b'generated once'


def test_retry_limit_is_pending_review_instead_of_a_model_failure(tmp_path):
    fixture(tmp_path)
    write_state(tmp_path,'semantic','001_v0',{'recovery_count':5})
    assert recover(tmp_path,'semantic','001_v0')['status']=='parked'
    write(tmp_path/'gold.json',{'001':{'composite':False}})
    protocol=read(tmp_path/'protocol.json')
    write(tmp_path/'protocol.json',{**protocol,'case_ids':['001'],'execution_methods':['semantic']})
    from tools.run_unified_md_comparison import summarize
    rows,metrics=summarize(tmp_path)
    assert rows[0]['transport_status']=='parked'
    assert metrics['semantic']['conditions_finished']==0
    assert not read(tmp_path/'progress.json')['complete']


def test_rag_selection_rounds_have_fresh_context_and_candidate_cache_keys():
    from comfyrouter.whole_request_selection import select_whole_request
    client=Mock()
    client.complete_json.return_value={'search_needed':True,'search_query':'text to image',
                                       'workflows':[],'reason':'missing'}
    for i in [1,2]:
        select_whole_request(client,[{'workflow_id':f'W{i}','description':f'candidate {i}'}],
            'original request','t2i',[],[],'semantic',anonymous_descriptions=True,
            retrieval_feedback=True,retrieval_round=i,retrieval_limit=3)
    payloads=[call.args[2] for call in client.complete_json.call_args_list]
    assert all(p['_prompt_history_mode']=='latest_only' for p in payloads)
    assert payloads[0]['_prompt_cache_namespace']!=payloads[1]['_prompt_cache_namespace']


def test_transport_retry_continues_same_selection_without_an_extra_retrieval(tmp_path,monkeypatch):
    import sys
    import types
    from tools import run_anonymous_md_comparison as router
    from comfyrouter.agentic import AgentTransportError
    dest=fixture(tmp_path)
    record=recover(tmp_path,'semantic','001_v0')
    checkpoint_path=tmp_path/'routing/001_v0/semantic_selection_checkpoint.json'
    checkpoint=read(checkpoint_path)
    checkpoint['retrieval_evidence']['query_count']=1
    checkpoint['retrieval_evidence']['rounds']=checkpoint['retrieval_evidence']['rounds'][-1:]
    checkpoint['retrieval_evidence']['rounds'][0]['round']=1
    write(checkpoint_path,checkpoint)
    write(tmp_path/'task_inputs.json',{'001_v0':{'instruction':'original request','modality':'t2i',
                                               'resources':[],'input_types':[]}})
    write(tmp_path/'workflow_catalog.json',[{'workflow_id':'W1','description':'generate image'}])
    protocol=read(tmp_path/'protocol.json');write(tmp_path/'protocol.json',{**protocol,'rag_top_k':20})
    encoder=Mock();encoder.encode.side_effect=AssertionError('must not retrieve again')
    monkeypatch.setattr(router,'configure',lambda _:None)
    monkeypatch.setattr(router,'client_type',lambda:Mock)
    monkeypatch.setattr('comfyrouter.semantic_retrieval.E5Encoder',lambda _:encoder)
    monkeypatch.setattr('numpy.load',lambda _:None)
    monkeypatch.setattr(router.time,'sleep',lambda _:None)
    from comfyrouter import whole_request_selection
    call=Mock(side_effect=AgentTransportError("Server error '502 Bad Gateway'"))
    monkeypatch.setattr(whole_request_selection,'select_whole_request',call)
    router.route_worker(tmp_path,'semantic','001_v0')
    route=read(tmp_path/'routing/001_v0/route.json')['methods']['semantic']
    assert route['status']=='agent_transport_failed'
    assert route['retrieval_evidence']['query_count']==1
    assert call.call_count==3
    assert len(route['transport_failures'])==3
    assert all(c.kwargs['retrieval_round']==1 for c in call.call_args_list)
