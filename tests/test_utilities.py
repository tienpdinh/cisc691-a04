import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from classes.utilities import delete_directory


class TestUtilities:
    
    def test_delete_directory_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test directory inside temp_dir
            test_dir = Path(temp_dir) / "test_directory"
            test_dir.mkdir()
            
            # Create some files inside
            (test_dir / "file1.txt").write_text("content1")
            (test_dir / "file2.txt").write_text("content2")
            
            # Create a subdirectory
            subdir = test_dir / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_text("content3")
            
            # Verify directory exists before deletion
            assert test_dir.exists()
            assert len(list(test_dir.rglob("*"))) == 4  # 3 files + 1 subdir
            
            # Delete the directory
            delete_directory(str(test_dir))
            
            # Verify directory no longer exists
            assert not test_dir.exists()
    
    def test_delete_directory_nonexistent(self):
        nonexistent_path = "/path/that/does/not/exist"
        
        # Should not raise an exception, just log a warning
        delete_directory(nonexistent_path)
    
    @patch('classes.utilities.shutil.rmtree')
    def test_delete_directory_permission_error(self, mock_rmtree):
        mock_rmtree.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            delete_directory("/some/path")
    
    @patch('classes.utilities.shutil.rmtree')
    def test_delete_directory_generic_error(self, mock_rmtree):
        mock_rmtree.side_effect = OSError("Some other error")
        
        with pytest.raises(OSError):
            delete_directory("/some/path")
    
    def test_delete_directory_with_readonly_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_directory"
            test_dir.mkdir()
            
            # Create a file
            test_file = test_dir / "readonly_file.txt"
            test_file.write_text("content")
            
            # Make file read-only (this might cause issues on Windows)
            test_file.chmod(0o444)
            
            try:
                # This should still work because shutil.rmtree handles permissions
                delete_directory(str(test_dir))
                assert not test_dir.exists()
            except PermissionError:
                # On some systems, this might still raise PermissionError
                # In that case, the test should verify the exception is raised
                with pytest.raises(PermissionError):
                    delete_directory(str(test_dir))
    
    def test_delete_directory_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "empty_directory"
            test_dir.mkdir()
            
            assert test_dir.exists()
            assert test_dir.is_dir()
            
            delete_directory(str(test_dir))
            
            assert not test_dir.exists()
    
    def test_delete_directory_with_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_directory"
            test_dir.mkdir()
            
            # Create a regular file
            regular_file = test_dir / "regular.txt"
            regular_file.write_text("content")
            
            # Create a symlink (if supported by the system)
            try:
                symlink_file = test_dir / "symlink.txt"
                symlink_file.symlink_to(regular_file)
                has_symlink = True
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system
                has_symlink = False
            
            delete_directory(str(test_dir))
            
            assert not test_dir.exists()
            if has_symlink:
                # Verify the original file (if it was outside the deleted directory) still exists
                # In this case, both the symlink and target were in the same directory
                assert not regular_file.exists()