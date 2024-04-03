import tensorflow as tf       # To work with USE4
import tensorflow_hub as hub  # contains USE4
from numpy import dot                                           # to calculate the dot product of two vectors
from numpy.linalg import norm      
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #Model is imported from this URL
model = hub.load(module_url)
def embed(input):
  return model(input)

def predict(text1, text2):
    # Convert the strings to lowercase for case-insensitive comparison
    messages = [text1, text2]
    message_embeddings = embed(messages)
    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))
    
    # Calculate similarity score and scale it to the range [0, 1]
    similarity_score = (cos_sim + 1) / 2
    return similarity_score
    
