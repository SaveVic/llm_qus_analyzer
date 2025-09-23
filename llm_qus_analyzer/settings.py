from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, TypeVar
from dotenv import load_dotenv
import yaml

T = TypeVar("T")


@dataclass
class LLMModelInfo:
    """Represents a Large Language Model (LLM) with its identifier and name."""

    id: str
    """A unique identifier for the LLM."""

    name: str
    """The display name of the LLM."""
    source:str 
    """source of the llm"""

@dataclass
class Configuration:
    """Stores configuration settings for the LLM application."""
    # Dihapus karena sudah tanpa meggunakan  apikey langchain sudah otomatis mengambil 
    # api_key: str | None = None
    # """The API key for accessing LLM services (optional)."""

    llm_models: list[LLMModelInfo] = field(default_factory=lambda: [])
    """List of available LLM models (defaults to empty list)."""


class Settings:
    """
    Manages application settings including environment variables and model configurations.

    This class handles loading and validating settings from environment files (.env)
    and model configuration files (YAML).
    """

    def __init__(self) -> None:
        """Initializes Settings with default configuration and loads default environment and model files."""
        self.__config = Configuration()
        self.__env_path = None
        self.__model_config_path = None

    def __load_environment(self, path: Path) -> None:
        """
        Load environment variables from .env file.

        Args:
            path (Path): Path to the .env file to load.

        Raises:
            Exception: Prints any exceptions that occur during loading.
        """
        try:
            if os.path.exists(path):
                load_dotenv(path, override=True)
                self.__env_path = path
            else:
                print(f"Environment file ({path}) not found.")
                load_dotenv()
            api_key = self.__get_env("TOGETHER_API_KEY", required=True)
            if api_key:
                self.__config.api_key = api_key
        except Exception as e:
            print(e.args[0])

    def __load_model_config(self, path: Path) -> None:
        """
        Load model configurations from YAML file.

        Args:
            path (Path): Path to the YAML configuration file to load.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                raw_model_config = yaml.safe_load(f)
            self.__model_config_path = path
            models = []
            if isinstance(raw_model_config, dict) and "models" in raw_model_config:
                model_list = raw_model_config["models"]
                if isinstance(model_list, list):
                    for model in model_list:
                        if isinstance(model, dict):
                            if "id" in model and "name" in model:
                                models.append(LLMModelInfo(model["id"], model["name"],model['source']))
            if len(models) > 0:
                self.__config.llm_models = deepcopy(models)
        else:
            print(f"Model configuration file ({path}) not found.")

    def __get_env(
        self, key: str, default: Optional[T] = None, required: bool = False
    ) -> T:
        """
        Get environment variable.

        Args:
            key (str): Name of the environment variable to retrieve.
            default (T | None): Default value to return if variable not found.
            required (bool): Whether the variable is required (raises ValueError if missing).

        Returns:
            value (T): The value of the environment variable or default.

        Raises:
            ValueError: If required variable is not set.
        """
        value = os.getenv(key, default)
        if required and value is None:
            raise ValueError(f"Required environment variable {key} not set")
        return value

    @property
    def config(self) -> Configuration:
        """
        Get the current configuration.

        Returns:
            config (Configuration): The current configuration object.

        Raises:
            KeyError: If required environment variables or model configurations are not set.
        """
        if self.__env_path is None or self.__config.api_key is None:
            raise KeyError("Environment variable is not set")
        if self.__model_config_path is None or len(self.__config.llm_models) == 0:
            raise KeyError("Model configuration is not set")
        return self.__config

    def configure_paths_and_load(
        self, env_path: Optional[Path] = None, model_config_path: Optional[Path] = None
    ) -> None:
        """
        Configure custom paths for environment and model configuration files and load them.

        Args:
            env_path (Path | None): Optional custom path to .env file.
            model_config_path (Path | None): Optional custom path to model configuration YAML file.
        """
        if env_path:
            self.__load_environment(env_path)
        if model_config_path:
            self.__load_model_config(model_config_path)
