# Testing Documentation

## Overview

This document describes the testing strategy and framework for LOOK-DGC, a digital image forensics toolkit.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # pytest configuration and fixtures
├── test_requirements.txt    # testing dependencies
├── unit/                    # Unit tests
│   ├── test_utility.py     # Tests for utility functions
│   ├── test_image_loading.py # Tests for image loading
│   └── test_histogram.py   # Tests for histogram analysis
├── integration/            # Integration tests
│   └── test_main_application.py # Main application tests
└── fixtures/               # Test data and fixtures
```

## Testing Framework

### Tools Used
- **pytest**: Main testing framework
- **pytest-qt**: Qt testing support
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Mocking utilities

### Key Features
- **Fixtures**: Reusable test data and setup
- **Parametrized tests**: Test multiple scenarios efficiently
- **Mocking**: Isolate units under test
- **Coverage reporting**: Track test coverage

## Writing Tests

### Unit Tests

Unit tests should:
- Test individual functions or methods
- Be fast and isolated
- Not depend on external systems
- Use mocks where appropriate

Example:
```python
def test_pad_image(sample_image):
    """Test image padding functionality"""
    original_shape = sample_image.shape
    padded = pad_image(sample_image, 16)
    
    assert padded.shape[0] >= original_shape[0]
    assert padded.shape[1] >= original_shape[1]
```

### Integration Tests

Integration tests should:
- Test interactions between components
- Verify system behavior
- May require more setup
- Test real-world scenarios

### Fixtures

Common fixtures available:
- `sample_image`: 100x100 RGB test image
- `sample_grayscale_image`: 100x100 grayscale image
- `sample_image_path`: Temporary image file path
- `mock_qt_app`: Mock Qt application

## Running Tests

### Basic Commands

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/unit/test_utility.py

# Run with coverage
python -m pytest --cov=gui --cov-report=html
```

### Test Categories

```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only  
python -m pytest tests/integration/

# Run tests matching pattern
python -m pytest -k "test_image"
```

## CI/CD Integration

### GitHub Actions Workflows

1. **basic-tests.yml**: Minimal testing for quick feedback
2. **tests.yml**: Comprehensive testing with coverage

### Automated Testing Triggers
- Push to main/develop branches
- Pull requests
- Scheduled runs (optional)

## Test Coverage

Aim for:
- **Unit tests**: 80%+ coverage for core utility functions
- **Integration tests**: Cover critical user workflows
- **Edge cases**: Test error conditions and boundary values

## Best Practices

### Test Design
- Follow AAA pattern (Arrange, Act, Assert)
- Use descriptive test names
- Keep tests independent
- Test one thing per test

### Code Quality
- Maintain test code quality
- Keep tests readable and maintainable
- Remove redundant tests
- Update tests when code changes

### Performance
- Keep unit tests fast (< 1 second each)
- Use appropriate fixtures for setup/teardown
- Mock expensive operations
- Parallelize when possible

## Adding New Tests

### Process
1. Identify what needs testing
2. Choose appropriate test type (unit/integration)
3. Write test using existing patterns
4. Run tests to verify
5. Update documentation if needed

### Example Test Addition

```python
# tests/unit/test_new_module.py
import pytest
from new_module import important_function

def test_important_function():
    """Test the important function with valid input"""
    result = important_function("test_input")
    assert result == "expected_output"
    
def test_important_function_edge_case():
    """Test edge case handling"""
    with pytest.raises(ValueError):
        important_function(None)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `gui/` directory is in Python path
2. **Qt errors**: Use `mock_qt_app` fixture for GUI tests
3. **Image file issues**: Use `tmp_path` fixture for temporary files
4. **Coverage not showing**: Check coverage configuration

### Debugging Tests

```bash
# Run single test with output
python -m pytest tests/unit/test_utility.py::test_pad_image -v -s

# Run with Python debugger
python -m pytest --pdb tests/unit/test_utility.py::test_pad_image

# Show test durations
python -m pytest --durations=10
```

## Future Improvements

Planned enhancements:
- Add performance benchmarks
- Implement property-based testing
- Add more integration tests for GUI components
- Include end-to-end test scenarios
- Automated visual regression testing