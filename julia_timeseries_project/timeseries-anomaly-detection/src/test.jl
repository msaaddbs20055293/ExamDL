using DataFrames
using CSV
using Flux
using JLD2
using Statistics

# Load the dataset
data = CSV.File("path/to/anomaly-free.csv"; delim=';') |> DataFrame

# Preprocess the data
# Assuming the dataset is already cleaned and ready for use
# Create labels (1 for anomaly-free, 0 for anomalies)
# For demonstration, let's assume we have a function to load anomaly data
# anomaly_data = load_anomaly_data() # Load your anomaly datasets here
# labels = vcat(ones(nrow(data)), zeros(nrow(anomaly_data))) # Create labels

# Create sliding windows
function create_windows(data, window_size)
    windows = []
    for i in 1:(nrow(data) - window_size + 1)
        push!(windows, data[i:(i + window_size - 1), :])
    end
    return windows
end

# Create windows for different sizes
windows_30 = create_windows(data, 30)
windows_90 = create_windows(data, 90)
windows_270 = create_windows(data, 270)

# Define the model
function create_model()
    return Chain(
        Dense(8, 64, relu),  # Input size is 8 (number of features)
        Dense(64, 32, relu),
        Dense(32, 1, σ)      # Output layer for binary classification
    )
end

# Training function
function train_model(model, train_data, labels, epochs, batch_size)
    opt = ADAMW(0.001, 0.001)  # AdamW optimizer
    for epoch in 1:epochs
        for i in 1:batch_size:length(train_data)
            x_batch = train_data[i:min(i + batch_size - 1, end)]
            y_batch = labels[i:min(i + batch_size - 1, end)]
            Flux.train!(loss, params(model), [(x_batch, y_batch)], opt)
        end
    end
end

# Save the model
function save_model(model, filename)
    JLD2.@save filename model
end

# Load the model
function load_model(filename)
    JLD2.@load filename model
    return model
end

# Calculate balanced accuracy
function bal_acc(predictions, labels)
    tp = sum((predictions .== 1) .& (labels .== 1))
    tn = sum((predictions .== 0) .& (labels .== 0))
    fp = sum((predictions .== 1) .& (labels .== 0))
    fn = sum((predictions .== 0) .& (labels .== 1))
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return (sensitivity + specificity) / 2
end

# Main execution
model = create_model()
train_model(model, windows_30, labels, 100, 128)  # Train with 30 time steps
save_model(model, "trained_model.jld2")

# Testing the model
# Load test data and labels
# test_data, test_labels = load_test_data()

# predictions = model(test_data)
# bal_accuracy = bal_acc(predictions, test_labels)
# println("Balanced Accuracy: ", bal_accuracy)