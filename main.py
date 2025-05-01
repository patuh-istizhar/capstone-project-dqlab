import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# --- Configuration ---
# Base directory where your saved_models folder is located.
# This variable defines the path to the folder where the trained machine learning models
# and their associated files (like the scaler) are stored.
# Ensure this path is correct relative to where you run the Streamlit app script.
# The correct path depends on the directory structure of your project and
# the location from which you execute the 'streamlit run your_app_name.py' command.
# Assuming you are running this script from the project root directory:
BASE_MODEL_SAVE_DIR = "saved_models"

# Dictionary mapping dataset names to their corresponding directories and data loaders
DATASET_CONFIG = {
    "Iris": {
        "dir": "iris_classification",
        "loader": load_iris,
        "scaling_required": [
            "Logistic Regression",
            "XGBoost",
        ],  # Models that were trained on scaled data
        "description": "Classify Iris flowers into Setosa, Versicolor, or Virginica based on measurements.",
        "input_method": ["Manual Input", "Upload CSV File"],  # Allowed input methods
        "emoji": "🌸",  # Emoji for the title and selectbox
        "display_name": "Iris Flower Classification 🌸",
        "class_colors": {  # Text Colors for styling prediction results
            "setosa": "color: green",
            "versicolor": "color: blue",
            "virginica": "color: red",
        },
    },
    "Breast Cancer": {
        "dir": "breast_cancer_classification",
        "loader": load_breast_cancer,
        "scaling_required": [
            "Logistic Regression",
            "XGBoost",
        ],  # Models that were trained on scaled data
        "description": "Predict if a breast tumor is Malignant or Benign based on cell measurements.",
        "input_method": ["Manual Input", "Upload CSV File"],  # Allowed input methods
        "emoji": "🎗️",  # Emoji for the title and selectbox
        "display_name": "Breast Tumor Prediction 🎗️",
        "class_colors": {  # Text Colors for styling prediction results
            "malignant": "color: red",
            "benign": "color: green",
        },
    },
    "Wine": {
        "dir": "wine_classification",
        "loader": load_wine,
        "scaling_required": [
            "Logistic Regression",
            "XGBoost",
        ],  # Models that were trained on scaled data
        "description": "Classify wines into different cultivars based on chemical analysis.",
        "input_method": ["Manual Input", "Upload CSV File"],  # Allowed input methods
        "emoji": "🍷",  # Emoji for the title and selectbox
        "display_name": "Wine Cultivar Analysis 🍷",
        "class_colors": {  # Text Colors for styling prediction results
            "class_0": "color: red",
            "class_1": "color: orange",
            "class_2": "color: purple",
        },
    },
}


# --- Helper Function to Load Model and Scaler ---
@st.cache_resource  # Cache the loaded resources to avoid reloading on every interaction
def load_resources(dataset_name):
    """Loads the best model, scaler, and configuration for the selected dataset."""
    config = DATASET_CONFIG[dataset_name]
    project_dir = os.path.join(BASE_MODEL_SAVE_DIR, config["dir"])

    # Load configuration file
    config_path = os.path.join(project_dir, "model_config.txt")
    loaded_config = {}
    try:
        with open(config_path, "r") as f:
            for line in f:
                key, value = line.strip().split(":", 1)
                loaded_config[key.strip()] = value.strip()

        # Parse specific config values
        best_model_name = loaded_config.get("Best_Model")
        requires_scaling = (
            loaded_config.get("Requires_Scaling", "False").lower() == "true"
        )

    except FileNotFoundError:
        st.error(
            f"Configuration file not found for {dataset_name} at {config_path}. Please ensure models are saved correctly."
        )
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading configuration for {dataset_name}: {e}")
        return None, None, None, None, None

    # Determine the expected model filename based on the best model name from config
    if best_model_name:
        model_filename = os.path.join(
            project_dir, f"{best_model_name.replace(' ', '_').lower()}_model.joblib"
        )
    else:
        st.error(f"Best model name not found in configuration for {dataset_name}.")
        return None, None, None, None, None

    # Load the model
    try:
        loaded_model = joblib.load(model_filename)
    except FileNotFoundError:
        st.error(
            f"Model file not found for {dataset_name} at {model_filename}. Please ensure models are saved correctly."
        )
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model for {dataset_name}: {e}")
        return None, None, None, None, None

    # Load the scaler (only if scaling is required or if a scaler file exists)
    loaded_scaler = None
    scaler_path = os.path.join(project_dir, "scaler.joblib")
    if os.path.exists(scaler_path):  # Check if scaler file exists
        try:
            loaded_scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(
                f"Scaler file found but could not be loaded for {dataset_name}: {e}"
            )

    # Get target names directly from the dataset loader (more reliable)
    try:
        data_loader = DATASET_CONFIG[dataset_name]["loader"]
        loaded_target_names = data_loader().target_names.tolist()
    except Exception as e:
        st.error(f"Error loading target names for {dataset_name}: {e}")
        loaded_target_names = []

    return (
        loaded_model,
        loaded_scaler,
        loaded_config,
        requires_scaling,
        loaded_target_names,
    )


# --- Streamlit App UI ---
st.set_page_config(
    page_title="Model Deployment App ✨", layout="wide"
)  # Set page title and layout

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")

# Get display names and sort them alphabetically
dataset_display_names = sorted(
    list(config["display_name"] for config in DATASET_CONFIG.values())
)

# Set default index to Iris Flower Classification
default_index = dataset_display_names.index(DATASET_CONFIG["Iris"]["display_name"])

# Use display names in the selectbox
selected_display_name = st.sidebar.selectbox(
    "Select App:",
    dataset_display_names,
    index=default_index,  # Set default to Iris
)

# Map the display name back to the internal dataset key
selected_dataset = next(
    key
    for key, config in DATASET_CONFIG.items()
    if config["display_name"] == selected_display_name
)


# Load resources based on selection
model, scaler, config_data, requires_scaling, target_names = load_resources(
    selected_dataset
)

# Proceed only if resources were loaded successfully
if model is not None and target_names is not None:
    st.subheader("Get Predictions")

    # Get the feature names for the selected dataset
    data_loader = DATASET_CONFIG[selected_dataset]["loader"]
    dummy_data = data_loader()
    feature_names = dummy_data.feature_names
    # Get descriptive statistics for input range hints
    dummy_df = pd.DataFrame(dummy_data.data, columns=feature_names)
    feature_description = dummy_df.describe().T  # Transpose for easier access

    # --- Input Method Selection ---
    # Define the options for the radio button
    input_options = ["Manual Input", "Upload CSV File"]

    # Determine the default selected index based on the dataset
    if selected_dataset == "Iris":
        default_input_method_index = input_options.index(
            "Manual Input"
        )  # Default to Manual for Iris
    else:
        default_input_method_index = input_options.index(
            "Upload CSV File"
        )  # Default to Upload CSV for others

    input_method = st.radio(
        "Choose Input Method:",
        input_options,  # Use the consistent options list
        index=default_input_method_index,  # Set default based on the dataset
    )

    input_df = None  # Initialize input_df
    uploaded_file = None  # Initialize uploaded_file outside the if block

    if input_method == "Manual Input":
        st.write("Enter the feature values below:")
        # Use columns to arrange input fields
        num_columns = 3  # Number of columns for input fields
        cols = st.columns(num_columns)

        input_data = {}
        for i, feature in enumerate(feature_names):
            with cols[i % num_columns]:  # Place input in the next column
                # Use a unique key for each input field to avoid issues with Streamlit reruns
                # Set a default value to avoid empty inputs
                # Add tooltip with min/max/mean from dataset description
                if feature in feature_description.index:
                    min_val = feature_description.loc[feature, "min"]
                    max_val = feature_description.loc[feature, "max"]
                    mean_val = feature_description.loc[feature, "mean"]
                    tooltip_text = f"Typical Range: {min_val:.2f} - {max_val:.2f} (Mean: {mean_val:.2f})"
                else:
                    tooltip_text = "No typical range available."

                input_data[feature] = st.number_input(
                    f"{feature}:",
                    key=f"{selected_dataset}_{feature}",
                    value=float(feature_description.loc[feature, "mean"])
                    if feature in feature_description.index
                    else 0.0,  # Default to mean if available
                    help=tooltip_text,  # Add tooltip
                )

        # Convert manual input to DataFrame
        input_df = pd.DataFrame([input_data])

    elif input_method == "Upload CSV File":
        st.write("Upload a CSV file containing the feature values.")

        st.info(
            f"Your CSV file should contain the following columns: **{', '.join(feature_names)}**. "
            f"Please fill in your data starting from the row **below** the headers. "  # Added instruction
            f"Each row should represent a data point for prediction. "
            f"Download a template below if needed. 📄"
        )

        # --- Download CSV Template Button ---
        @st.cache_data  # Cache the template data
        def generate_csv_template(features):
            template_df = pd.DataFrame(columns=features)
            return template_df.to_csv(index=False).encode("utf-8")

        csv_template = generate_csv_template(
            feature_names
        )  # Pass feature_names to the function

        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name=f"{selected_dataset.lower().replace(' ', '_')}_template.csv",
            mime="text/csv",
            help="Download a CSV file with the correct column headers.",
        )
        st.markdown("---")  # Separator after download button

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        # Display warning if no file is uploaded
        if uploaded_file is None:
            st.warning("Please upload a CSV file to proceed with prediction. ⬆️")
        else:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                # Hide the index column in the preview
                st.dataframe(input_df.head(), hide_index=True)

                # Optional: Basic validation of columns
                uploaded_cols = set(input_df.columns)
                expected_cols = set(feature_names)
                if not expected_cols.issubset(uploaded_cols):
                    missing_cols = list(expected_cols - uploaded_cols)
                    st.warning(
                        f"Warning: Uploaded file is missing expected feature columns: {missing_cols} ⚠️"
                    )
                    # Decide whether to proceed or stop. For now, we'll proceed but warn.
                if not uploaded_cols.issuperset(expected_cols):
                    extra_cols = list(uploaded_cols - expected_cols)
                    st.warning(
                        f"Warning: Uploaded file contains unexpected columns: {extra_cols}. These will be ignored. 🤔"
                    )

            except Exception as e:
                st.error(f"Error reading CSV file: {e} ❌")
                input_df = None  # Reset input_df if there's an error

    st.markdown("---")  # Add a separator

    # Prediction button - only enabled if input_df is not None (meaning data is ready)
    if st.button("✨ Get Prediction ✨", disabled=(input_df is None)):
        # Apply scaling if required for the loaded model and scaler is available
        processed_input = input_df.copy()  # Start with a copy of the input data

        if requires_scaling and scaler is not None:
            try:
                # Ensure input_df has the correct columns in the correct order for scaling
                # This is important if the uploaded CSV columns are in a different order
                processed_input = scaler.transform(input_df[feature_names])
            except Exception as e:
                st.sidebar.error(f"Error applying scaler: {e} ❌")
                processed_input = None  # Prevent prediction if scaling fails
        elif requires_scaling and scaler is None:
            st.sidebar.warning(
                "Model requires scaling, but scaler was not loaded. Prediction may be inaccurate. ⚠️"
            )
            # Decide whether to proceed with unscaled data or stop.
            # For now, let's proceed but keep the warning prominent.
            # Ensure input_df has the correct columns in the correct order
            processed_input = input_df[
                feature_names
            ].values  # Convert to numpy array if not scaled
        else:  # No scaling required
            # Ensure input_df has the correct columns in the correct order
            processed_input = input_df[feature_names].values  # Convert to numpy array

        if processed_input is not None:
            # Make prediction
            try:
                predictions = model.predict(processed_input)

                st.subheader("Prediction Results:")

                if input_method == "Manual Input":
                    # For manual input, display a single prediction
                    predicted_class_index = predictions[0]
                    if 0 <= predicted_class_index < len(target_names):
                        predicted_class_name = target_names[predicted_class_index]
                        st.success(f"Predicted Class: **{predicted_class_name}**! 🎉")
                    else:
                        # This case should be rare now that target_names are loaded directly
                        st.warning(
                            f"Model predicted an unknown class index: {predicted_class_index} 🤔"
                        )
                        st.info(f"Raw prediction output: {predictions}")

                elif input_method == "Upload CSV File":
                    # For file upload, display predictions for each row in a styled table
                    # Map numerical predictions back to target names
                    predicted_class_names = [
                        target_names[int(p)]
                        if 0 <= int(p) < len(target_names)
                        else "Unknown"
                        for p in predictions
                    ]

                    # Create a results DataFrame that includes original features and predictions
                    results_df = input_df.copy()
                    results_df["Predicted_Class"] = predicted_class_names

                    # --- Apply styling to the Predicted_Class column ---
                    def color_predicted_class(row):
                        # Get the color mapping for the current dataset
                        colors = DATASET_CONFIG[selected_dataset].get(
                            "class_colors", {}
                        )
                        # Normalize class name for dictionary lookup (handle spaces and case)
                        predicted_class_key = (
                            row["Predicted_Class"].lower().replace(" ", "_")
                        )
                        # Change from background-color to color for text color
                        style = colors.get(
                            predicted_class_key, "color: black"
                        )  # Get text color style, default to black
                        return [
                            style if col == "Predicted_Class" else ""
                            for col in row.index
                        ]

                    # Display the full DataFrame with styled predictions, hiding the index
                    st.dataframe(
                        results_df.style.apply(color_predicted_class, axis=1),
                        hide_index=True,
                    )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e} 🐛")

else:
    st.warning(
        "Could not load model resources. Please check the console for errors and ensure models are saved correctly. ⚠️"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by **Patuh Istizhar**")
st.sidebar.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/patuh)")
st.sidebar.markdown(
    "[View Source on GitHub](https://github.com/patuh-istizhar/capstone-project-dqlab)"
)
