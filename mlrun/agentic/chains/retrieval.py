# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.config import (
    get_embedding_function,
    get_object_from_dict,
    get_vector_db,
    vector_db_shortcuts,
)
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.agentic.utils import logger


class DocumentCallbackHandler(BaseCallbackHandler):
    def on_retriever_end(self, documents: List[Document], **kwargs):
        logger.debug("Retrieved documents", documents=documents)
        for i, doc in enumerate(documents):
            doc.metadata["index"] = str(i)


class DocumentRetriever:
    def __init__(
        self,
        llm,
        vector_store,
        verbose: bool = False,
        chain_type: Optional[str] = None,
        **search_kwargs,
    ):
        document_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {index}",
            input_variables=["page_content", "index"],
        )

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs=search_kwargs),
            chain_type=chain_type or "stuff",
            return_source_documents=True,
            chain_type_kwargs={"document_prompt": document_prompt},
            verbose=verbose,
        )
        self.cb = DocumentCallbackHandler()
        self.cb.verbose = verbose
        self.verbose = verbose
        self.chain_type = chain_type

    @classmethod
    def from_dicts(
        cls,
        llm_args: dict,
        vector_store_args: dict,
        embeddings_args: dict,
        collection_name: Optional[str] = None,
        verbose: bool = False,
        **search_kwargs,
    ):
        """Create a DocumentRetriever from plain config dicts.

        :param llm_args:          Dict with ``class_name`` + LLM kwargs.
        :param vector_store_args: Dict with ``class_name`` + vector store kwargs.
        :param embeddings_args:   Dict with ``class_name`` + embeddings kwargs.
        :param collection_name:   Override collection name.
        :param verbose:           Enable verbose logging.
        """
        from mlrun.agentic.config import get_llm

        vector_db = get_vector_db(vector_store_args, embeddings_args, collection_name=collection_name)
        llm = get_llm(llm_args)
        return cls(llm, vector_db, verbose=verbose, **search_kwargs)

    def _get_answer(self, query: str) -> tuple[str, List[Document]]:
        result = self.chain({"question": query}, callbacks=[self.cb])
        sources = [s.strip() for s in result["sources"].split(",")]
        source_docs = [
            doc
            for doc in result["source_documents"]
            if doc.metadata.pop("index", "") in sources
        ]
        if self.verbose:
            docs_string = "\n".join(str(doc.metadata) for doc in source_docs)
            logger.info("Source documents", docs=docs_string)
        return result["answer"], source_docs

    def run(self, event: WorkflowEvent) -> Dict[str, any]:
        logger.debug("Retriever question", query=event.query)
        query = event.query.content if hasattr(event.query, "content") else event.query
        answer, sources = self._get_answer(query)
        logger.debug("Retriever result", answer=answer, sources=sources)
        return {"answer": answer, "sources": sources}


class MultiRetriever(ChainRunner):
    """RAG retrieval chain that creates DocumentRetrievers per collection.

    :param model_name:        LLM model name (default gpt-4).
    :param temperature:       LLM temperature (default 0).
    :param llm:               Pre-built LLM (overrides model_name/temperature).
    :param default_collection: Default vector store collection name.
    :param vector_store_args:  Dict with ``class_name`` + connection args for the
                               vector store (e.g. ``{"class_name": "milvus", ...}``).
    :param embeddings_args:    Dict with ``class_name`` + args for embeddings
                               (e.g. ``{"class_name": "huggingface", ...}``).
    """

    def __init__(
        self,
        model_name="gpt-4",
        temperature=0,
        llm=None,
        default_collection="default",
        vector_store_args=None,
        embeddings_args=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._temperature = temperature
        self._llm = llm
        self.default_collection = default_collection
        self._vector_store_args = vector_store_args or {
            "class_name": "milvus",
            "collection_name": "default",
            "connection_args": {"address": "localhost:19530"},
        }
        self._embeddings_args = embeddings_args or {
            "class_name": "huggingface",
            "model_name": "all-MiniLM-L6-v2",
        }
        self._retrievers: Dict[str, DocumentRetriever] = {}

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        if not self._llm:
            self._llm = ChatOpenAI(
                model=self._model_name, temperature=self._temperature
            )

    def _get_vector_db(self, collection_name):
        embeddings = get_embedding_function(self._embeddings_args)
        vs_args = self._vector_store_args.copy()
        if collection_name:
            vs_args["collection_name"] = collection_name
        vs_args["embedding_function"] = embeddings
        return get_object_from_dict(vs_args, vector_db_shortcuts)

    def _get_retriever(
        self, collection_name: Optional[str] = None
    ) -> DocumentRetriever:
        collection_name = collection_name or self.default_collection
        logger.debug("Selected collection", collection_name=collection_name)
        if collection_name not in self._retrievers:
            vector_db = self._get_vector_db(collection_name)
            retriever = DocumentRetriever(self._llm, vector_db, verbose=self.verbose)
            self._retrievers[collection_name] = retriever
        return self._retrievers[collection_name]

    def _run(self, event: WorkflowEvent) -> Dict[str, any]:
        collection_name = event.kwargs.get("collection_name")
        retriever = self._get_retriever(collection_name)
        return retriever.run(event)


def fix_milvus_filter_arg(vector_db, search_kwargs: Dict[str, any]):
    if "filter" in search_kwargs and hasattr(vector_db, "_create_connection_alias"):
        filter_arg = search_kwargs.pop("filter")
        if isinstance(filter_arg, dict):
            filter_str = " and ".join(f"{k}={v}" for k, v in filter_arg.items())
        else:
            filter_str = filter_arg
        search_kwargs["expr"] = filter_str


def get_retriever_from_dicts(
    llm_args: dict,
    vector_store_args: dict,
    embeddings_args: dict,
    verbose: bool = False,
    collection_name: Optional[str] = None,
    **search_kwargs,
) -> DocumentRetriever:
    """Create a DocumentRetriever from plain config dicts."""
    return DocumentRetriever.from_dicts(
        llm_args=llm_args,
        vector_store_args=vector_store_args,
        embeddings_args=embeddings_args,
        collection_name=collection_name,
        verbose=verbose,
        **search_kwargs,
    )
