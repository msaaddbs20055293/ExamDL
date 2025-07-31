using DataFrames
using CSV
using Flux
using JLD2
using Random

# Load the dataset
data = CSV.File("path/to/anomaly-free.csv"; delim=';') |> DataFrame

# Preprocess the data
# Convert datetime to a usable format if necessary
data.datetime = DateTime.(data.datetime)

# Normalize the features
features = select(data, Not(:datetime))
features .= (features .- mean(features, dims=1)) ./ std(features, dims=1)

# Create time series windows
function create_windows(data, window_size)
    X, y = [], []
    for i in 1:(nrow(data) - window_size)
        push!(X, Matrix(data[i:(i + window_size - 1), Not(:target)])')
        push!(y, data.target[i + window_size - 1])
    end
    return X, y
end

# Assuming 'target' is the binary label column
data.target = ... # Define your target variable based on the anomaly detection

# Create windows for different sizes
X_30, y_30 = create_windows(features, 30)
X_90, y_90 = create_windows(features, 90)
X_270, y_270 = create_windows(features, 270)

# Define the model
function create_model()
    model = Chain(
        Dense(30, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 1, σ)  # Output layer for binary classification
    )
    return model
end

# Training function
function train_model(X, y, model)
    opt = ADAM(0.001, 0.001)  # Adam optimizer with learning rate and weight decay
    loss(x, y) = Flux.Losses.binarycrossentropy(model(x), y)
    
    for epoch in 1:100
        for i in 1:128:length(X)
            x_batch = X[i:min(i + 127, end)]
            y_batch = y[i:min(i + 127, end)]
            Flux.train!(loss, params(model), [(x_batch, y_batch)], opt)
        end
    end
end

# Train models for different window sizes
model_30 = create_model()
train_model(X_30, y_30, model_30)

model_90 = create_model()
train_model(X_90, y_90, model_90)

model_270 = create_model()
train_model(X_270, y_270, model_270)

# Save the models
JLD2.@save "model_30.jld2" model_30
JLD2.@save "model_90.jld2" model_90
JLD2.@save "model_270.jld2" model_270

# Evaluation function
function evaluate_model(model, X_test, y_test)
    predictions = model(X_test)
    return Flux.Losses.binarycrossentropy(predictions, y_test)
end

# Load the model for evaluation
JLD2.@load "model_30.jld2" model_30
# Assuming X_test and y_test are prepared similarly to the training data
bal_acc = evaluate_model(model_30, X_test, y_test)
println("Balanced Accuracy: ", bal_acc)