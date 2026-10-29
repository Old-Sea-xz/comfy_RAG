import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comfyrouter.semantic_retrieval import SemanticWorkflowIndex

class Encoder:

    def encode(self, texts, query=False):
        return np.array([[1.0, 0.0] for _ in texts])

class RetrievalTests(unittest.TestCase):

    def documents(self):
        return [{'workflow_id': m + '/DifferentFamily/op/model', 'modality': m, 'task_family': 'DifferentFamily', 'operation': 'DifferentFamily/op', 'selected_profile': 'model', 'base_path': '/frozen/' + m} for m in ['t2i', 'i2i', 'v2v']]

    def test_filter_only_media_not_semantic_family(self):
        index = SemanticWorkflowIndex(self.documents(), [[1, 0], [0.5, 0.5], [1, 0]], Encoder())
        self.assertEqual([r['modality'] for r in index.search('anything', 'i2i')], ['i2i'])
        self.assertEqual(index.search('anything', 'i2i')[0]['task_family'], 'DifferentFamily')

    def test_missing_contract_stops_instead_of_fallback(self):
        index = SemanticWorkflowIndex(self.documents(), [[1, 0]] * 3, Encoder())
        with self.assertRaisesRegex(ValueError, 'no workflow'):
            index.search('anything', 'i2v')

    def test_one_shot_blocks_json_retry_without_sending(self):
        from comfyrouter.agentic import OpenAICompatibleClient, AgentOutputError
        from tools.run_routing_comparison import client_type
        cls = client_type()
        client = cls.__new__(cls)
        client.calls = set()
        with patch.object(OpenAICompatibleClient, '_complete_json') as send:
            with self.assertRaises(AgentOutputError):
                client._complete_json('a', {}, {'_json_retry': 1})
            send.assert_not_called()

    def test_second_decision_blocked_but_distinct_stages_allowed(self):
        from comfyrouter.agentic import OpenAICompatibleClient, AgentOutputError
        from tools.run_routing_comparison import client_type
        cls = client_type()
        client = cls.__new__(cls)
        client.calls = set()
        with patch.object(OpenAICompatibleClient, '_complete_json', return_value={}) as send:
            for step in ['1', '2']:
                payload = {'implementation_plan': [{'operation': 'F/op', 'decomposition_step': {'step': step}}]}
                client._complete_json('generation_agent', {}, payload)
            with self.assertRaises(AgentOutputError):
                client._complete_json('generation_agent', {}, payload)
            self.assertEqual(send.call_count, 2)
if __name__ == '__main__':
    unittest.main()
