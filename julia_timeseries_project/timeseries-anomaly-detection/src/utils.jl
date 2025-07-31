using CSV
using DataFrames
using Flux
using JLD2
using Statistics

# Load the dataset
data = CSV.File("path/to/anomaly-free.csv"; delim=';')
df = DataFrame(data)

# Preprocess the data
# Assume df has columns: datetime, Accelerometer1RMS, ..., Volume Flow RateRMS
# Create labels (0 for anomaly-free, 1 for anomalies)
# This is a placeholder; you need to load your anomaly data accordingly
labels = [0 for _ in 1:size(df, 1)]  # Replace with actual labels for anomalies

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = split_data(df, labels)

# Define the model
function create_model(input_size)
    return Chain(
        Dense(input_size, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 1, σ)  # Output layer with sigmoid activation for binary classification
    )
end

# Train the model
function train_model(model, train_x, train_y)
    opt = ADAMW(0.001, 0.001)
    for epoch in 1:100
        Flux.train!(loss, params(model), [(train_x, train_y)], opt)
    end
end

# Evaluate the model
function evaluate_model(model, test_x, test_y)
    predictions = model(test_x)
    # Calculate balanced accuracy
    bal_acc = calculate_balanced_accuracy(predictions, test_y)
    return bal_acc
end

# Save the model
function save_model(model, filename)
    JLD2.@save filename model
end

# Main execution
model = create_model(input_size)
train_model(model, train_x, train_y)
bal_acc = evaluate_model(model, test_x, test_y)
save_model(model, "trained_model.jld2")

# test.jl
# Load the model and evaluate on test data
JLD2.@load "trained_model.jld2" trained_model
bal_acc = evaluate_model(trained_model, test_x, test_y)
println("Balanced Accuracy: ", bal_acc)