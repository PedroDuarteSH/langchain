from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class LlamaCppEmbeddings(BaseModel, Embeddings):
    """llama.cpp embedding models.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LlamaCppEmbeddings
            llama = LlamaCppEmbeddings(model_path="/path/to/model.bin")
    """

    client: Any = None  #: :meta private:
    """The Llama client object."""
    
    model_path: Optional[str] = Field(default=None, alias="model_path")
    """Path to the Llama model file."""
    
    repo_id: Optional[str] = Field(default=None, alias="repo_id")
    """Repository ID for the model, if using from_pretrained method."""
    
    filename: Optional[str] = Field(default=None, alias="filename")
    """Filename for the model, if using from_pretrained method."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(False, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use. If None, the number 
    of threads is automatically determined."""

    n_batch: Optional[int] = Field(512, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    verbose: bool = Field(True, alias="verbose")
    """Print verbose output to stderr."""

    device: Optional[str] = Field(None, alias="device")
    """Device type to use and pass to the model"""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

     

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that llama-cpp-python library is installed."""
        model_path = self.model_path
        model_param_names = [
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "verbose",
            "device",
            "filename"
        ]
        model_params = {k: getattr(self, k) for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if self.n_gpu_layers is not None:
            model_params["n_gpu_layers"] = self.n_gpu_layers

        if not self.client:
            if not model_path and not self.repo_id:
                raise ValueError(
                    "You must provide either a model_path or a repo_id to load a LlamaCPP embeddings model."
                )
            if model_path and self.repo_id:
                raise ValueError("Only one of model_path or repo_id can be provided, not both.")
            
            try:
                if self.repo_id:
                    from llama_cpp import Llama
                    self.client = Llama.from_pretrained(self.repo_id, embedding=True, **model_params)
                if model_path:
                    from llama_cpp import Llama
                    self.client = Llama(model_path, embedding=True, **model_params)
            except ImportError:
                raise ImportError(
                    "Could not import llama-cpp-python library. "
                    "Please install the llama-cpp-python library to "
                    "use this embedding model: pip install llama-cpp-python"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load Llama model from path: {model_path} or repo_id: {self.repo_id}. "
                    f"Received error {e}"
                )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self.client.create_embedding(texts)
        

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.client.embed(text)
        
