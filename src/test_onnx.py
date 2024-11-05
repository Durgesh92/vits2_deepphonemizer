import onnx

# Load the ONNX model
model_path = "vits.onnx"  # Replace with the actual path to your model file
model = onnx.load(model_path)

# Get all node names (similar to keys)
node_names = [node.name for node in model.graph.node]
print("Node names (keys):", node_names[:10])  # Display the first 10 node names as a sample
