import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from classes.embedding_preparer import EmbeddingPreparer


class TestEmbeddingPreparer:
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_init(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            preparer = EmbeddingPreparer(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            assert preparer.file_list == ["test.txt"]
            assert preparer.input_dir == input_dir
            assert preparer.output_dir == output_dir
            assert preparer.embedding_model_name == "test-model"
            assert preparer.device == torch.device("cpu")
            assert output_dir.exists()
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_init_with_cuda(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = True
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            preparer = EmbeddingPreparer(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            assert preparer.device == torch.device("cuda")
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_read_file(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_content = "  Test content  "
            test_file.write_text(test_content, encoding="utf-8")
            
            preparer = EmbeddingPreparer(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = preparer._read_file(test_file)
            assert result == "Test content"
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_generate_embedding(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Mock tokenizer output
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        mock_inputs_with_device = Mock()
        mock_inputs_with_device.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_with_device
        
        # Mock model output
        mock_tensor = Mock()
        mock_tensor.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = mock_tensor
        mock_model.return_value = mock_outputs
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            preparer = EmbeddingPreparer(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = preparer._generate_embedding("test text")
            assert result == [0.1, 0.2, 0.3]
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_save_embedding(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            preparer = EmbeddingPreparer(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            test_file_path = Path("test.txt")
            test_embedding = [0.1, 0.2, 0.3]
            
            preparer._save_embedding(test_file_path, test_embedding)
            
            output_file = output_dir / "test_embeddings.json"
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                saved_embedding = json.load(f)
            assert saved_embedding == test_embedding
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_process_files_success(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Mock tokenizer and model for embedding generation
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_inputs_with_device = Mock()
        mock_inputs_with_device.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_with_device
        
        mock_tensor = Mock()
        mock_tensor.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = mock_tensor
        mock_model.return_value = mock_outputs
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_file.write_text("Test content", encoding="utf-8")
            
            preparer = EmbeddingPreparer(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            preparer.process_files()
            
            output_file = output_dir / "test_embeddings.json"
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                saved_embedding = json.load(f)
            assert saved_embedding == [0.1, 0.2, 0.3]
    
    @patch('classes.embedding_preparer.AutoModel')
    @patch('classes.embedding_preparer.AutoTokenizer')
    @patch('classes.embedding_preparer.torch.cuda.is_available')
    def test_process_files_file_not_found(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        mock_cuda_available.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            preparer = EmbeddingPreparer(
                file_list=["nonexistent.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            # Should not raise an exception, just log a warning
            preparer.process_files()
            
            # No output file should be created
            output_file = output_dir / "nonexistent_embeddings.json"
            assert not output_file.exists()