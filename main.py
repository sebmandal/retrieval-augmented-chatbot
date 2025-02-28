from rag_starter import main as rs

if input("Do you want to train the model? (n): ").lower() == "y":
    training_file = "data/input.txt"
    embeddings_json = rs.train(training_file)

thread_id = rs.open_thread()

rs.send_message("Hello, how are you?", thread_id)

# Note: Should return at least: "Since seldom coming in that long year set,"
messages = rs.send_message(
    "What is the entire sonnet that contains: 'Therefore are feasts so solemn and so rare,'?",
    thread_id)

for message in messages:
    print(message["role"], message["content"])
