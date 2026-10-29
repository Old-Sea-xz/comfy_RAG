import unittest
from unittest.mock import Mock
from comfyrouter.whole_request_selection import select_whole_request
from comfyrouter.agentic import AgentOutputError
from tools.run_anonymous_md_comparison import execution_attempt_budget


class WholeRequestSelectionTests(unittest.TestCase):
    def test_retrieval_does_not_reduce_the_three_execution_attempt_budget(self):
        self.assertEqual([execution_attempt_budget(n) for n in (1, 2, 3)], [3, 3, 3])


    def documents(self):
        return [dict(workflow_id='edit',workflow_path='/edit.json',base_path='/base',
            modality='i2i',operation='Object_Editing/removal',task_family='Object_Editing',
            selected_profile='model',description='Remove an object.',node_types=['LoadImage']),
            dict(workflow_id='upscale',workflow_path='/upscale.json',base_path='/base',
            modality='i2i',operation='Geometry_Editing/upscaling',task_family='Geometry_Editing',
            selected_profile='model',description='Upscale an image.',node_types=['LoadImage'])]

    def row(self,workflow_id,deps):
        return dict(workflow_id=workflow_id,instruction='Apply requested edit.',input_type='image',
            input_types=['image'],output_type='image',depends_on=deps,mask_role=None,
            mask_construction=None,target_presence=None,geometry_requirement=None,protected_regions=[])

    def test_one_request_jointly_selects_chain_without_a_gate_or_per_stage_calls(self):
        client=Mock();client.complete_json.return_value={'workflows':[self.row('edit',[]),
            self.row('upscale',[1])],'reason':'Remove then upscale.'}
        decision,spec,stages,raw=select_whole_request(client,self.documents(),
            'Remove the fork then upscale by 2x.','i2i',['source.png'],['image'],'direct')
        self.assertEqual(client.complete_json.call_count,1)
        payload=client.complete_json.call_args.args[2]
        self.assertEqual(payload['instruction'],'Remove the fork then upscale by 2x.')
        self.assertNotIn('decomposition',payload)
        self.assertNotIn('steps',payload)
        self.assertEqual(stages[1]['decomposition_step']['depends_on'],['1'])
        self.assertEqual([s['workflow_id'] for s in stages],['edit','upscale'])

    def test_unknown_candidates_and_forward_dependencies_are_rejected_without_retry(self):
        for row in [self.row('invented',[]),self.row('edit',[2])]:
            client=Mock();client.complete_json.return_value={'workflows':[row],'reason':'x'}
            with self.assertRaises(AgentOutputError):
                select_whole_request(client,self.documents(),'request','i2i',[],[],'semantic')
            self.assertEqual(client.complete_json.call_count,1)

    def test_mask_consumers_require_an_actual_declared_producer(self):
        client=Mock();row=self.row('edit',[]);row['input_types']=['image','mask']
        client.complete_json.return_value={'workflows':[row],'reason':'edit masked region'}
        with self.assertRaisesRegex(AgentOutputError,'requires a mask'):
            select_whole_request(client,self.documents(),'Edit the background.','i2i',
                                 ['source.png'],['image'],'semantic')

    def test_anonymous_mode_only_exposes_ids_and_descriptions_and_maps_execution(self):
        docs=self.documents()
        docs[0]['workflow_id']='W0123456789'
        docs[1]['workflow_id']='W9876543210'
        client=Mock();client.complete_json.return_value={'workflows':[self.row(docs[0]['workflow_id'],[])],'reason':'x'}
        _,_,stages,_=select_whole_request(client,docs,'Remove a fork.','i2i',
            ['source.png'],['image'],'direct',anonymous_descriptions=True)
        payload=client.complete_json.call_args.args[2]
        self.assertEqual(payload['candidates'],[{k:d[k] for k in ['workflow_id','description']} for d in docs])
        self.assertNotIn('overall_modality',payload)
        self.assertEqual(stages[0]['workflow_path'],'/edit.json')
        self.assertEqual(stages[0]['operation'],'Object_Editing/removal')

    def test_bare_semantic_profile_has_no_natural_language_routing_rules(self):
        docs=self.documents()
        docs[0]['workflow_id']='W0123456789'
        docs[1]['workflow_id']='W9876543210'
        client=Mock();client.complete_json.return_value={
            'workflows':[self.row(docs[0]['workflow_id'],[])], 'reason':'x'}
        select_whole_request(client,docs,'Remove a fork.','i2i',['source.png'],['image'],
            'direct_bare',anonymous_descriptions=True,prompt_profile='bare_semantic')
        agent_id,contract,payload=client.complete_json.call_args.args
        self.assertEqual(agent_id,'direct_semantic_selection_agent')
        self.assertEqual(contract,{
            'role':'Given the user request and the workflow descriptions, select suitable workflows and combine them.',
            'rules':[]})
        self.assertEqual(payload['instruction'],'Remove a fork.')
        self.assertEqual(payload['candidates'],[
            {k:d[k] for k in ['workflow_id','description']} for d in docs])
        self.assertNotIn('overall_modality',payload)
        self.assertIn('_response_json_schema',payload)

    def test_standard_profile_keeps_existing_prompt_rules(self):
        client=Mock();client.complete_json.return_value={
            'workflows':[self.row('edit',[])], 'reason':'x'}
        select_whole_request(client,self.documents(),'Remove a fork.','i2i',[],[],
            'direct',anonymous_descriptions=True)
        contract=client.complete_json.call_args.args[1]
        self.assertTrue(contract['rules'])
        self.assertIn('MASK outputs', ' '.join(contract['rules']))

    def test_unknown_prompt_profile_is_rejected_before_call(self):
        client=Mock()
        with self.assertRaisesRegex(ValueError,'unsupported whole-request prompt profile'):
            select_whole_request(client,self.documents(),'request','i2i',[],[],'direct',
                prompt_profile='unknown')
        client.complete_json.assert_not_called()

    def test_dependency_with_wrong_media_type_is_rejected(self):
        client=Mock();first=self.row('edit',[]);second=self.row('upscale',[1])
        first['output_type']='mask'
        client.complete_json.return_value={'workflows':[first,second],'reason':'x'}
        with self.assertRaises(ValueError):
            select_whole_request(client,self.documents(),'request','i2i',[],[],'direct')

    def test_anonymous_infeasible_candidates_can_block_without_forced_selection(self):
        client=Mock();client.complete_json.return_value={'workflows':[],'reason':'No supplied template generates an image from text.'}
        with self.assertRaisesRegex(AgentOutputError,'selection blocked:'):
            select_whole_request(client,self.documents(),'Generate an image.','t2i',[],[],
                                 'semantic',anonymous_descriptions=True)
        schema=client.complete_json.call_args.args[2]['_response_json_schema']['schema']
        self.assertEqual(schema['properties']['workflows']['minItems'],0)
        self.assertEqual(client.complete_json.call_count,1)

    def test_retrieval_feedback_requests_only_a_missing_capability_query(self):
        client=Mock();client.complete_json.return_value={
            'search_needed':True,'search_query':'general text-to-image scene generation',
            'workflows':[],'reason':'No candidate accepts text alone to create an image.'}
        decision,spec,stages,raw=select_whole_request(client,self.documents(),
            'Generate a dog image.','t2i',[],[],'semantic',anonymous_descriptions=True,
            retrieval_feedback=True,retrieval_round=1,retrieval_limit=3)
        self.assertIsNone(decision);self.assertIsNone(spec);self.assertIsNone(stages)
        self.assertTrue(raw['search_needed'])
        payload=client.complete_json.call_args.args[2]
        self.assertEqual(payload['stage'],'anonymous_description_selection_round_1')
        schema=payload['_response_json_schema']['schema']
        self.assertEqual(set(schema['required']),{'search_needed','search_query','workflows','reason'})

    def test_retrieval_feedback_requires_empty_query_when_candidates_are_sufficient(self):
        client=Mock();client.complete_json.return_value={
            'search_needed':False,'search_query':'','workflows':[self.row('edit',[])],
            'reason':'The candidate directly supports the requested edit.'}
        _,_,stages,raw=select_whole_request(client,self.documents(),'Remove a fork.','i2i',[],[],
            'semantic',anonymous_descriptions=True,retrieval_feedback=True,
            retrieval_round=2,retrieval_limit=3)
        self.assertEqual(stages[0]['workflow_id'],'edit')
        self.assertFalse(raw['search_needed'])

    def test_retrieval_feedback_rejects_missing_query_and_nonempty_final_query(self):
        bad=[{'search_needed':True,'search_query':'','workflows':[],'reason':'missing'},
             {'search_needed':False,'search_query':'more terms','workflows':[self.row('edit',[])],
              'reason':'sufficient'}]
        for response in bad:
            client=Mock();client.complete_json.return_value=response
            with self.assertRaises(AgentOutputError):
                select_whole_request(client,self.documents(),'request','i2i',[],[],'semantic',
                    anonymous_descriptions=True,retrieval_feedback=True,retrieval_round=3,retrieval_limit=3)


if __name__=='__main__':unittest.main()
