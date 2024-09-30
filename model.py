import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set HuggingFace API key and repo ID
sec_key = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize the generative model from HuggingFace
llm_gen = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.7,
    token=sec_key
)

# 1. Alloy Composition Generation
alloy_gen_template = '''
Generate an optimal alloy composition for {material_type} used in {application}.
Alloy Composition:
'''
prompt_alloy = PromptTemplate(
    input_variables=['material_type', 'application'],
    template=alloy_gen_template
)

# 2. Material Property Prediction
property_pred_template = '''
Predict material properties for the following alloy composition:
Composition: {composition}
Material Properties:
'''
prompt_property = PromptTemplate(
    input_variables=['composition'],
    template=property_pred_template
)

# 3. Corrosion Resistance Prediction
corrosion_resistance_template = '''
Predict the corrosion resistance of the following material in {environment}:
Material: {material}
Corrosion Resistance:
'''
prompt_corrosion = PromptTemplate(
    input_variables=['material', 'environment'],
    template=corrosion_resistance_template
)

# 4. Heat Treatment Outcome Prediction
heat_treatment_template = '''
Predict the outcome of heat treatment for the following material at {temperature}°C with a cooling rate of {cooling_rate}°C/s:
Material: {material}
Outcome:
'''
prompt_heat_treatment = PromptTemplate(
    input_variables=['material', 'temperature', 'cooling_rate'],
    template=heat_treatment_template
)

# Create Streamlit App UI
st.title("MetAIxpert- AI Expert for Material Science and Metallurgy")

st.sidebar.title("Choose a Metallurgical Task")
task = st.sidebar.selectbox(
    "Task",
    (
        "Alloy Composition Generation",
        "Material Property Prediction",
        "Corrosion Resistance Prediction",
        "Heat Treatment Outcome Prediction"
    ),
)

# Alloy Composition Generation
if task == "Alloy Composition Generation":
    st.header("Alloy Composition Generation")
    material_type = st.text_input("Enter material type (e.g., Steel, Aluminum):")
    application = st.text_input("Enter application (e.g., aerospace, automotive):")
    if st.button("Generate Alloy Composition"):
        if material_type and application:
            alloy_chain = LLMChain(llm=llm_gen, prompt=prompt_alloy)
            response = alloy_chain.run({"material_type": material_type, "application": application})
            st.write("### Suggested Alloy Composition:")
            st.write(response)
        else:
            st.write("Please provide material type and application.")

# Material Property Prediction
elif task == "Material Property Prediction":
    st.header("Material Property Prediction")
    composition = st.text_area("Enter alloy composition (e.g., Fe-98%, C-2%):")
    if st.button("Predict Material Properties"):
        if composition:
            property_chain = LLMChain(llm=llm_gen, prompt=prompt_property)
            response = property_chain.run({"composition": composition})
            st.write("### Predicted Material Properties:")
            st.write(response)
        else:
            st.write("Please enter an alloy composition.")

# Corrosion Resistance Prediction
elif task == "Corrosion Resistance Prediction":
    st.header("Corrosion Resistance Prediction")
    material = st.text_input("Enter material (e.g., Stainless Steel, Copper):")
    environment = st.selectbox("Select Environment:", ["Marine", "Industrial", "Tropical"])
    if st.button("Predict Corrosion Resistance"):
        if material:
            corrosion_chain = LLMChain(llm=llm_gen, prompt=prompt_corrosion)
            response = corrosion_chain.run({"material": material, "environment": environment})
            st.write("### Corrosion Resistance Prediction:")
            st.write(response)
        else:
            st.write("Please provide material and environment.")

# Heat Treatment Outcome Prediction
elif task == "Heat Treatment Outcome Prediction":
    st.header("Heat Treatment Outcome Prediction")
    material = st.selectbox("Select Material:", ["Steel", "Aluminum", "Copper"])
    temperature = st.slider("Select Heat Treatment Temperature (°C):", 200, 1200)
    cooling_rate = st.slider("Select Cooling Rate (°C/s):", 0.1, 10.0)
    if st.button("Predict Outcome"):
        if material:
            heat_treatment_chain = LLMChain(llm=llm_gen, prompt=prompt_heat_treatment)
            response = heat_treatment_chain.run({"material": material, "temperature": temperature, "cooling_rate": cooling_rate})
            st.write("### Heat Treatment Outcome:")
            st.write(response)
        else:
            st.write("Please provide material, temperature, and cooling rate.")

# Footer and Contact Info
st.sidebar.info("Developed by Rachit Ranjan.")
