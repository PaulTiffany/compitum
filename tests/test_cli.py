import runpy
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import yaml


def test_cli_route_command_verbose(tmp_path: Any, capsys: Any) -> None:
    """Test the CLI's route command with the --verbose flag."""
    from compitum.cli import main
    # Arrange
    defaults_path = tmp_path / "router_defaults.yaml"
    defaults_path.write_text(yaml.dump({
        "metric": {"D": 35, "rank": 4, "delta": 0.001},
        "alpha": 0.4, "beta_t": 0.2, "beta_c": 0.15, "beta_d": 0.15, "beta_s": 0.1,
        "update_stride": 8
    }))
    constraints_path = tmp_path / "constraints_us_default.yaml"
    constraints_path.write_text(yaml.dump({"A": [[1,0,0,0]], "b": [1]}))

    with patch('compitum.cli.CalibratedPredictor', MagicMock()), \
         patch('compitum.cli.CompitumRouter') as mock_CompitumRouter:

        mock_router_instance = MagicMock()
        mock_cert = MagicMock()
        mock_cert.to_json.return_value = '{"model": "mock_model"}'
        mock_router_instance.route.return_value = mock_cert
        mock_CompitumRouter.return_value = mock_router_instance

        # Act
        test_args = ["compitum", "route", "--prompt", "test prompt", "--verbose",
                     "--defaults", str(defaults_path), "--constraints", str(constraints_path)]
        with patch.object(sys, 'argv', test_args):
            main()

        # Assert
        mock_CompitumRouter.assert_called_once()
        mock_router_instance.route.assert_called_with("test prompt")
        mock_cert.to_json.assert_called_once()
        captured = capsys.readouterr()
        assert '{"model": "mock_model"}' in captured.out

def test_cli_route_command_non_verbose(tmp_path: Any, capsys: Any) -> None:
    """Test the CLI's route command without the --verbose flag."""
    from compitum.cli import main
    # Arrange
    defaults_path = tmp_path / "router_defaults.yaml"
    defaults_path.write_text(yaml.dump({
        "metric": {"D": 35, "rank": 4, "delta": 0.001},
        "alpha": 0.4, "beta_t": 0.2, "beta_c": 0.15, "beta_d": 0.15, "beta_s": 0.1,
        "update_stride": 8
    }))
    constraints_path = tmp_path / "constraints_us_default.yaml"
    constraints_path.write_text(yaml.dump({"A": [[1,0,0,0]], "b": [1]}))

    with patch('compitum.cli.CalibratedPredictor', MagicMock()), \
         patch('compitum.cli.CompitumRouter') as mock_CompitumRouter:

        mock_router_instance = MagicMock()
        mock_cert = MagicMock()
        mock_cert.model = "mock_model"
        mock_cert.utility = 0.9
        mock_router_instance.route.return_value = mock_cert
        mock_CompitumRouter.return_value = mock_router_instance

        # Act
        test_args = ["compitum", "route", "--prompt", "test prompt",
                     "--defaults", str(defaults_path), "--constraints", str(constraints_path)]
        with patch.object(sys, 'argv', test_args):
            main()

        # Assert
        captured = capsys.readouterr()
        assert '"model": "mock_model"' in captured.out
        assert '"U": 0.9' in captured.out

def test_cli_main_entrypoint() -> None:
    """
    Tests the if __name__ == '__main__' block in cli.py
    """
    mock_safe_load = MagicMock(side_effect=[
        { # First call for router_defaults.yaml
            "metric": {"D": 35, "rank": 1, "delta": 1}, "update_stride": 1,
            "alpha": 1, "beta_t": 1, "beta_c": 1, "beta_d": 1, "beta_s": 1
        },
        { # Second call for constraints_us_default.yaml
            "A": [[1, 0, 0, 0]],
            "b": [2.0]
        }
    ])

    # Unload the module if it was imported by other tests
    if 'compitum.cli' in sys.modules:
        del sys.modules['compitum.cli']

    mock_router_class = MagicMock()
    mock_router_instance = mock_router_class.return_value
    mock_cert = mock_router_instance.route.return_value
    mock_cert.model = "mock_model_name"
    mock_cert.utility = 0.8

    with patch('pathlib.Path.read_text', return_value="---"), \
         patch('compitum.router.CompitumRouter', mock_router_class), \
         patch('yaml.safe_load', mock_safe_load), \
         patch('sys.argv', ["cli.py", "route", "--prompt", "test"]):

        runpy.run_module('compitum.cli', run_name='__main__')

