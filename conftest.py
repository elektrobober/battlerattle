# Makes the repo root importable so tests can `import dnd_pipeline`.


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: медленные тесты (компиляция PDF)")
