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

"""Factory helpers for creating LLMs, embeddings, and vector stores from dicts.

Each ``get_*`` function accepts a plain dict with a ``class_name`` key
plus constructor kwargs.  Short aliases (e.g. ``"huggingface"``,
``"milvus"``) are expanded automatically.
"""

import importlib


embeddings_shortcuts = {
    "huggingface": "langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings",
    "openai": "langchain_openai.embeddings.base.OpenAIEmbeddings",
}

vector_db_shortcuts = {
    "milvus": "langchain_community.vectorstores.Milvus",
    "chroma": "langchain_community.vectorstores.chroma.Chroma",
}

llm_shortcuts = {
    "chat": "langchain_openai.ChatOpenAI",
    "gpt": "langchain_community.chat_models.GPT",
}


def get_embedding_function(embeddings_args: dict):
    """Create an embeddings instance from a dict.

    :param embeddings_args: Dict with ``class_name`` + constructor kwargs.
    """
    return get_object_from_dict(embeddings_args, embeddings_shortcuts)


def get_llm(llm_args: dict):
    """Create an LLM instance from a dict.

    :param llm_args: Dict with ``class_name`` + constructor kwargs.
    """
    return get_object_from_dict(llm_args, llm_shortcuts)


def get_vector_db(
    vector_store_args: dict,
    embeddings_args: dict,
    collection_name: str = None,
):
    """Create a vector store instance from dicts.

    :param vector_store_args: Dict with ``class_name`` + connection kwargs.
    :param embeddings_args:   Dict with ``class_name`` + kwargs for embeddings.
    :param collection_name:   Override the collection name in vector_store_args.
    """
    embeddings = get_embedding_function(embeddings_args)
    vector_store_args = vector_store_args.copy()
    if collection_name:
        vector_store_args["collection_name"] = collection_name
    vector_store_args["embedding_function"] = embeddings
    return get_object_from_dict(vector_store_args, vector_db_shortcuts)


def get_class_from_string(class_path, shortcuts: dict = {}) -> type:
    if class_path in shortcuts:
        class_path = shortcuts[class_path]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_object_from_dict(obj_dict: dict, shortcuts: dict = {}):
    if not isinstance(obj_dict, dict):
        return obj_dict
    obj_dict = obj_dict.copy()
    class_name = obj_dict.pop("class_name")
    class_ = get_class_from_string(class_name, shortcuts)
    return class_(**obj_dict)
