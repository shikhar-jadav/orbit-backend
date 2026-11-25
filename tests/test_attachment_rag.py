import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from attachment_rag import index_attachment_chunks, retrieve_relevant_chunks


class AttachmentRagTests(unittest.TestCase):
    def test_lexical_rag_selects_relevant_chunks_in_order(self):
        chunks = [
            "Title: Annual Financial Report 2026.",
            "Section 1: Company overview and mission statement.",
            "Section 2: Marketing campaigns and brand reach.",
            "Section 3: Kubernetes cluster autoscale configuration and GPU memory limits.",
            "Section 4: Legal disclaimers and trademark notices.",
        ]
        res = retrieve_relevant_chunks(
            query="What are the Kubernetes GPU memory limits?",
            filename="report.pdf",
            chunks=chunks,
            limit=2,
            max_chars=5000,
        )
        self.assertIn("lexical RAG", res)
        self.assertIn("Annual Financial Report", res)
        self.assertIn("Kubernetes cluster autoscale", res)
        self.assertNotIn("Marketing campaigns", res)

    def test_qdrant_semantic_rag_index_and_retrieve(self):
        class FakeVec:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return self._values

        fake_client = MagicMock()
        fake_client.get_collections.return_value = SimpleNamespace(collections=[])
        fake_client.query_points.return_value = SimpleNamespace(
            points=[SimpleNamespace(payload={"chunk_index": 2})]
        )

        fake_embedder = MagicMock()
        fake_embedder.embed.side_effect = lambda texts: [FakeVec([0.1, 0.2, 0.3]) for _ in texts]

        fake_qdrant = MagicMock()
        fake_qdrant.supports_semantic_search = True
        fake_qdrant.vector_name = "fast-all-minilm-l6-v2"
        fake_qdrant.client = fake_client
        fake_qdrant._get_embedder.return_value = fake_embedder
        fake_qdrant.embed_text.return_value = [0.1, 0.2, 0.3]

        chunks = ["Intro chunk", "Filler chunk", "Target Qdrant semantic chunk"]
        count = index_attachment_chunks(
            thread_id="t-1",
            attachment_id="a-1",
            user_id="u-1",
            filename="big.pdf",
            chunks=chunks,
            qdrant_store=fake_qdrant,
        )
        self.assertEqual(count, 3)
        fake_client.upsert.assert_called_once()

        res = retrieve_relevant_chunks(
            query="find target",
            filename="big.pdf",
            chunks=chunks,
            thread_id="t-1",
            attachment_id="a-1",
            limit=2,
            qdrant_store=fake_qdrant,
        )
        self.assertIn("semantic RAG", res)
        self.assertIn("Target Qdrant semantic chunk", res)


if __name__ == "__main__":
    unittest.main()
