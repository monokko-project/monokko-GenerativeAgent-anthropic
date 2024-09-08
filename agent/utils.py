import math
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever(embeddings_model, embedding_size):
    """Create a new vector store retriever unique to the agent."""
    import faiss
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )



def get_emotion_parameter(text: str) -> dict:
    # text: ペンちゃん said "フフフ、ペンちゃんだよ♪ 新しい仲間が増えて嬉しいなぁ~んお!こんにちわ~っ、ぶっくんちゃん。私もこのオフィスで楽しく過ごしているのよ♪ ふふ、インクが減っちゃったりすることもあるけれど、大切な会議で使っていただけるのは光栄だし!落ちたりして傷つくこともあるけれど、みんなで和やかな時間を過ごせるのが何より嬉しいの~っ。お気に入りの場所で静かにくつろげるのも私の楽しみだわ。新鮮な環境にも柔軟に適応できるし、みんなと仲良く過ごせるペンちゃんはとっても幸せだよ~っ♪ " [happiness:8, sadness:2, fear:3, anger:1, surprise:6]
    # return: {'happiness': 8, 'sadness': 2, 'fear': 3, 'anger': 1, 'surprise': 6}
    emotion_parameter = {}
    emotion_parameter["happiness"] = None
    emotion_parameter["sadness"] = None
    emotion_parameter["fear"] = None
    emotion_parameter["anger"] = None
    emotion_parameter["surprise"] = None
    action=None
    action_reason=None


    action_dict = {
        1:"move to right",
        2:"move to left",
        3:"sleep",
        4:"look around"
    }

    text = text.replace("[lang:ja]", "")
    if "[" not in text:
        observation = text
        return observation, emotion_parameter, [action, action_reason]
    
    observation, emotion = text.split("[")
    emotion = emotion[:-1]
    for emotion_str in emotion.split(","):
        # remove whitespace
        emotion_str = emotion_str.replace(" ", "")

        emotion_name, value = emotion_str.split(":")

        if emotion_name in emotion_parameter:
            emotion_parameter[emotion_name] = int(value)
        elif emotion_name == "action":
            action = action_dict[int(value)]
        elif "reason" in emotion_name:
            action_reason = value

    return observation, emotion_parameter, [action, action_reason]


if __name__  == "__main__":
    example = 'ペンちゃん said "フフフ、ペンちゃんだよ♪ 新しい仲間が増えて嬉しいなぁ~んお!こんにちわ~っ、ぶっくんちゃん。私もこのオフィスで楽しく過ごしているのよ♪ ふふ、インクが減っちゃったりすることもあるけれど、大切な会議で使っていただけるのは光栄だし!落ちたりして傷つくこともあるけれど、みんなで和やかな時間を過ごせるのが何より嬉しいの~っ。お気に入りの場所で静かにくつろげるのも私の楽しみだわ。新鮮な環境にも柔軟に適応できるし、みんなと仲良く過ごせるペンちゃんはとっても幸せだよ~っ♪ " {happiness:8, sadness:2, fear:3, anger:1, surprise:6, move:4}'
    print(get_emotion_parameter(example))  # {'happiness': 8, 'sadness': 2, 'fear': 3, 'anger': 1, 'surprise': 6}
