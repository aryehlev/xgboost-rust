use xgboost_rust::{Booster, XGBoostResult};

fn main() -> XGBoostResult<()> {
    println!("XGBoost Rust Bindings - Advanced Usage Example");
    println!("===============================================\n");

    // Load model
    let model_path = "iris_model.json";
    println!("Loading model from: {}", model_path);

    let booster = match Booster::load(model_path) {
        Ok(b) => {
            println!("✓ Model loaded successfully\n");
            b
        }
        Err(e) => {
            eprintln!("✗ Failed to load model: {}", e);
            eprintln!("\nPlease create a model file first. See basic_usage.rs for instructions.");
            return Err(e);
        }
    };

    // Example: Binary classification prediction
    println!("=== Example 1: Normal Prediction ===");
    let data = vec![
        5.1, 3.5, 1.4, 0.2,
        6.7, 3.0, 5.2, 2.3,
    ];

    let predictions = booster.predict(&data, 2, 4, 0, false)?;
    println!("Normal predictions (probabilities): {:?}\n", predictions);

    // Example: Get SHAP values (feature contributions)
    println!("=== Example 2: Feature Contributions (SHAP) ===");
    use xgboost_rust::predict_option;

    let shap_values = booster.predict(
        &data,
        2,
        4,
        predict_option::PRED_CONTRIBS,
        false
    )?;

    println!("SHAP values for first sample:");
    // SHAP values include one extra value for bias term
    let num_features = 4;
    for i in 0..=num_features {
        if i < num_features {
            println!("  Feature {}: {:.4}", i, shap_values[i]);
        } else {
            println!("  Bias term: {:.4}", shap_values[i]);
        }
    }
    println!();

    // Example: Save model to a new location
    println!("=== Example 3: Save Model ===");
    let save_path = "iris_model_copy.json";
    booster.save(save_path)?;
    println!("✓ Model saved to: {}\n", save_path);

    // Example: Load from buffer (useful for embedded models)
    println!("=== Example 4: Load from Buffer ===");
    let buffer = std::fs::read(model_path).map_err(|e| {
        xgboost_rust::XGBoostError {
            description: format!("Failed to read model file: {}", e),
        }
    })?;
    let booster_from_buffer = Booster::load_from_buffer(&buffer)?;
    println!("✓ Model loaded from buffer ({} bytes)", buffer.len());

    let num_features = booster_from_buffer.num_features()?;
    println!("  Model has {} features\n", num_features);

    // Make a prediction with the buffer-loaded model
    let test_data = vec![5.1, 3.5, 1.4, 0.2];
    let pred = booster_from_buffer.predict(&test_data, 1, 4, 0, false)?;
    println!("Prediction from buffer-loaded model: {:?}\n", pred);

    println!("Advanced examples completed successfully!");

    Ok(())
}
