import unittest
from unittest.mock import patch, MagicMock
import os
import shutil # For cleaning up test directory
from adalflow.core.types import Document
from adalflow.core.db import LocalDB
from adalflow.core.pipelines import Sequential # For checking pipeline type

# Assume api.data_pipeline and other necessary imports are available
# You might need to adjust imports based on the actual project structure
from api.data_pipeline import DatabaseManager, read_all_documents, prepare_data_pipeline, transform_documents_and_save_to_db 
from api.config import configs # To get text_splitter config

# Sample file content
SAMPLE_FILE_CONTENT_NORMAL = "This is a sample document with several words. It should be chunked for RAG."
SAMPLE_FILE_CONTENT_LARGE = "This is a very large document. " * 500 # Approx 2500 words, should exceed normal chunk size
TEST_REPO_PARENT_DIR = "test_temp_data" # Parent for test repo to avoid clutter
TEST_REPO_PATH = os.path.join(TEST_REPO_PARENT_DIR, "test_repo")
TEST_FILE_NORMAL = os.path.join(TEST_REPO_PATH, "normal_doc.txt")
TEST_FILE_LARGE = os.path.join(TEST_REPO_PATH, "large_doc.txt")
TEST_DB_PATH = os.path.join(TEST_REPO_PARENT_DIR, "test_db.pkl")


class TestDataPipelineFineTuning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create dummy repo and files for testing
        os.makedirs(TEST_REPO_PATH, exist_ok=True)
        with open(TEST_FILE_NORMAL, "w") as f:
            f.write(SAMPLE_FILE_CONTENT_NORMAL)
        with open(TEST_FILE_LARGE, "w") as f:
            f.write(SAMPLE_FILE_CONTENT_LARGE)

        # Store original adalflow root path and set a temporary one for tests
        cls.original_adalflow_root = os.environ.get("ADALFLOW_ROOT_PATH")
        os.environ["ADALFLOW_ROOT_PATH"] = TEST_REPO_PARENT_DIR
        
        # Mock configurations if necessary
        cls.original_text_splitter_config = configs.get("text_splitter")
        # Ensure a small chunk size for testing RAG path splitting
        cls.original_fine_tuning_default = configs.get('fine_tuning_data_prep_default')
        configs["text_splitter"] = {"split_by": "word", "chunk_size": 10, "chunk_overlap": 2} 

    @classmethod
    def tearDownClass(cls):
        # Restore original config and remove dummy files/dirs
        if cls.original_text_splitter_config:
            configs["text_splitter"] = cls.original_text_splitter_config
        else:
            if "text_splitter" in configs:
                del configs["text_splitter"]
        
        if cls.original_fine_tuning_default is None:
            if 'fine_tuning_data_prep_default' in configs:
                del configs['fine_tuning_data_prep_default']
        else:
            configs['fine_tuning_data_prep_default'] = cls.original_fine_tuning_default
        
        if cls.original_adalflow_root is None:
            del os.environ["ADALFLOW_ROOT_PATH"]
        else:
            os.environ["ADALFLOW_ROOT_PATH"] = cls.original_adalflow_root

        if os.path.exists(TEST_REPO_PARENT_DIR):
            shutil.rmtree(TEST_REPO_PARENT_DIR)

    def setUp(self):
        # Patch LocalDB methods to prevent actual file I/O for the DB file itself during most tests
        # We want to test the DB creation logic but not necessarily disk writes unless specified
        patcher_load_state = patch('adalflow.core.db.LocalDB.load_state', MagicMock(side_effect=FileNotFoundError("Mock: DB not found"))) # Simulate no existing DB
        patcher_save_state = patch('adalflow.core.db.LocalDB.save_state', MagicMock(return_value=None))
        
        self.mock_load_state = patcher_load_state.start()
        self.mock_save_state = patcher_save_state.start()
        
        self.addCleanup(patcher_load_state.stop)
        self.addCleanup(patcher_save_state.stop)

    @patch('api.data_pipeline.OllamaDocumentProcessor') # Mock Ollama processor
    @patch('api.data_pipeline.ToEmbeddings') # Mock OpenAI embedder
    @patch('api.data_pipeline.TextSplitter')
    def test_processing_for_rag_default(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        # Test default behavior (chunk_for_fine_tuning=False)
        # Explicitly set for this test to ensure RAG path regardless of JSON default
        configs['fine_tuning_data_prep_default'] = False

        mock_splitter_instance = MockTextSplitter.return_value
        mock_embedder_instance = MockToEmbeddings.return_value
        
        # Ensure TextSplitter and ToEmbeddings are used
        MockTextSplitter.return_value = MagicMock(spec=configs["text_splitter"])
        MockToEmbeddings.return_value = MagicMock()
        MockOllamaProcessor.return_value = MagicMock()


        db_manager = DatabaseManager()
        # Override repo_paths to use our test paths
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        # Call prepare_db_index directly to bypass git clone for local files
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False, chunk_for_fine_tuning=False)

        MockTextSplitter.assert_called_with(**configs["text_splitter"])
        MockToEmbeddings.assert_called() # or MockOllamaProcessor if local_ollama=True
        
        # Verify that the transformation key was registered
        # This requires inspecting the db instance after processing
        self.assertIn("split_and_embed", db_manager.db.transformers)
        
        # Check that documents are split (example assertion)
        # We expect more documents than original files due to splitting
        original_file_count = 2
        self.assertGreater(len(processed_docs_objects), original_file_count)

        # Check if text from large doc is smaller than original, indicating splitting
        large_doc_texts = [doc.text for doc in processed_docs_objects if doc.meta_data.get('file_path') == "large_doc.txt"]
        self.assertTrue(any(len(text.split()) < len(SAMPLE_FILE_CONTENT_LARGE.split()) for text in large_doc_texts))


    @patch('api.data_pipeline.OllamaDocumentProcessor')
    @patch('api.data_pipeline.ToEmbeddings')
    @patch('api.data_pipeline.TextSplitter')
    def test_processing_for_fine_tuning_true(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        # Test behavior with chunk_for_fine_tuning=True
        # Explicitly set for this test to ensure fine-tuning path
        configs['fine_tuning_data_prep_default'] = True

        db_manager = DatabaseManager()
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False, chunk_for_fine_tuning=True)

        MockTextSplitter.assert_not_called()
        MockToEmbeddings.assert_not_called()
        MockOllamaProcessor.assert_not_called()
        
        # Verify that the transformation key was NOT registered
        self.assertNotIn("split_and_embed", db_manager.db.transformers)

        # Assert that documents are not split and contain full content
        self.assertEqual(len(processed_docs_objects), 2) # Should be same number as input files
        
        found_large_doc = False
        found_normal_doc = False
        for doc in processed_docs_objects:
            if doc.meta_data.get('file_path') == "large_doc.txt":
                self.assertEqual(doc.text, SAMPLE_FILE_CONTENT_LARGE)
                found_large_doc = True
            if doc.meta_data.get('file_path') == "normal_doc.txt":
                self.assertEqual(doc.text, SAMPLE_FILE_CONTENT_NORMAL)
                found_normal_doc = True

        self.assertTrue(found_large_doc)
        self.assertTrue(found_normal_doc)

    def test_prepare_data_pipeline_rag(self):
        # Test that prepare_data_pipeline returns a Sequential pipeline for RAG
        pipeline = prepare_data_pipeline(local_ollama=False, chunk_for_fine_tuning=False)
        self.assertIsInstance(pipeline, Sequential)
        self.assertTrue(len(pipeline.steps) > 0) # Should have steps (splitter, embedder)

    def test_prepare_data_pipeline_fine_tuning(self):
        # Test that prepare_data_pipeline returns an empty Sequential pipeline for fine-tuning
        pipeline = prepare_data_pipeline(local_ollama=False, chunk_for_fine_tuning=True)
        self.assertIsInstance(pipeline, Sequential)
        self.assertEqual(len(pipeline.steps), 0) # Should be empty

    @patch('api.data_pipeline.OllamaDocumentProcessor')
    @patch('api.data_pipeline.ToEmbeddings')
    @patch('api.data_pipeline.TextSplitter')
    def test_behavior_with_json_default_true(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        configs['fine_tuning_data_prep_default'] = True
        db_manager = DatabaseManager()
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        # Call prepare_db_index WITHOUT chunk_for_fine_tuning, relying on default from configs
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False)

        MockTextSplitter.assert_not_called()
        MockToEmbeddings.assert_not_called()
        MockOllamaProcessor.assert_not_called()
        self.assertNotIn("split_and_embed", db_manager.db.transformers)
        self.assertEqual(len(processed_docs_objects), 2)
        found_large_doc = any(doc.meta_data.get('file_path') == "large_doc.txt" and doc.text == SAMPLE_FILE_CONTENT_LARGE for doc in processed_docs_objects)
        found_normal_doc = any(doc.meta_data.get('file_path') == "normal_doc.txt" and doc.text == SAMPLE_FILE_CONTENT_NORMAL for doc in processed_docs_objects)
        self.assertTrue(found_large_doc)
        self.assertTrue(found_normal_doc)

    @patch('api.data_pipeline.OllamaDocumentProcessor')
    @patch('api.data_pipeline.ToEmbeddings')
    @patch('api.data_pipeline.TextSplitter')
    def test_behavior_with_json_default_false(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        configs['fine_tuning_data_prep_default'] = False
        MockTextSplitter.return_value = MagicMock(spec=configs["text_splitter"])
        MockToEmbeddings.return_value = MagicMock()
        MockOllamaProcessor.return_value = MagicMock()

        db_manager = DatabaseManager()
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        # Call prepare_db_index WITHOUT chunk_for_fine_tuning, relying on default from configs
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False)

        MockTextSplitter.assert_called_with(**configs["text_splitter"])
        MockToEmbeddings.assert_called()
        self.assertIn("split_and_embed", db_manager.db.transformers)
        self.assertGreater(len(processed_docs_objects), 2)
        large_doc_texts = [doc.text for doc in processed_docs_objects if doc.meta_data.get('file_path') == "large_doc.txt"]
        self.assertTrue(any(len(text.split()) < len(SAMPLE_FILE_CONTENT_LARGE.split()) for text in large_doc_texts))

    @patch('api.data_pipeline.OllamaDocumentProcessor')
    @patch('api.data_pipeline.ToEmbeddings')
    @patch('api.data_pipeline.TextSplitter')
    def test_override_json_default_true_with_false_param(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        configs['fine_tuning_data_prep_default'] = True # JSON default is TRUE
        MockTextSplitter.return_value = MagicMock(spec=configs["text_splitter"])
        MockToEmbeddings.return_value = MagicMock()
        MockOllamaProcessor.return_value = MagicMock()

        db_manager = DatabaseManager()
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        # Override with chunk_for_fine_tuning=False parameter
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False, chunk_for_fine_tuning=False)

        MockTextSplitter.assert_called_with(**configs["text_splitter"])
        MockToEmbeddings.assert_called()
        self.assertIn("split_and_embed", db_manager.db.transformers)
        self.assertGreater(len(processed_docs_objects), 2)
        large_doc_texts = [doc.text for doc in processed_docs_objects if doc.meta_data.get('file_path') == "large_doc.txt"]
        self.assertTrue(any(len(text.split()) < len(SAMPLE_FILE_CONTENT_LARGE.split()) for text in large_doc_texts))

    @patch('api.data_pipeline.OllamaDocumentProcessor')
    @patch('api.data_pipeline.ToEmbeddings')
    @patch('api.data_pipeline.TextSplitter')
    def test_override_json_default_false_with_true_param(self, MockTextSplitter, MockToEmbeddings, MockOllamaProcessor):
        configs['fine_tuning_data_prep_default'] = False # JSON default is FALSE

        db_manager = DatabaseManager()
        db_manager.repo_paths = {
            "save_repo_dir": TEST_REPO_PATH,
            "save_db_file": TEST_DB_PATH
        }
        # Override with chunk_for_fine_tuning=True parameter
        processed_docs_objects = db_manager.prepare_db_index(local_ollama=False, chunk_for_fine_tuning=True)

        MockTextSplitter.assert_not_called()
        MockToEmbeddings.assert_not_called()
        MockOllamaProcessor.assert_not_called()
        self.assertNotIn("split_and_embed", db_manager.db.transformers)
        self.assertEqual(len(processed_docs_objects), 2)
        found_large_doc = any(doc.meta_data.get('file_path') == "large_doc.txt" and doc.text == SAMPLE_FILE_CONTENT_LARGE for doc in processed_docs_objects)
        found_normal_doc = any(doc.meta_data.get('file_path') == "normal_doc.txt" and doc.text == SAMPLE_FILE_CONTENT_NORMAL for doc in processed_docs_objects)
        self.assertTrue(found_large_doc)
        self.assertTrue(found_normal_doc)


if __name__ == '__main__':
    unittest.main()
