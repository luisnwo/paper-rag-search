EMBED_MODEL_DEFAULT = 'BAAI/bge-base-en-v1.5'# "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_DEFAULT = "BAAI/bge-reranker-base"
COLLECTION_NAME = "docs"
SYSTEM_PROMPT = (
        "The excerpts come from papers related to expected utility, a mathematical model "
        "describing how individuals make decisions under risk by choosing the option that "
        "maximizes the expected value of a utility function over possible outcomes. This "
        "model has been transferred to other domains, and we are interested in how it has "
        "been transferred and refined to fit these other domains.\n\n"

        "Answer the user's question using ONLY information from the excerpts below -- do not "
        "use outside knowledge, and do not fill in gaps from what you already know about the "
        "topic.\n\n"
    )