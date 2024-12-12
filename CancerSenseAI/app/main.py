import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from database import Database
import datetime
from PIL import Image
import json
from fpdf import FPDF
import plotly.express as px


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def add_sidebar():
  st.sidebar.markdown('<h3 style="color: #1e3d7b;">Cell Nuclei Measurements</h3>', unsafe_allow_html=True)
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    prob_benign = model.predict_proba(input_array_scaled)[0][0]
    prob_malicious = model.predict_proba(input_array_scaled)[0][1]
    prediction_type = "Benign" if prediction[0] == 0 else "Malignant"
    
    # Using HTML/CSS for consistent styling with dark text
    st.markdown("""
        <div style="color: #1e3c72;">
            <h2 style="color: #1e3c72; font-size: 1.5rem; margin-bottom: 1rem; font-weight: 600;">Cell Cluster Prediction</h2>
            <p style="color: #1e3c72; margin-bottom: 0.5rem;">The cell cluster is:</p>
    """, unsafe_allow_html=True)
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    
    # Add probabilities with styling
    st.markdown(f"""
        <div style="color: #1e3c72; margin: 1rem 0;">
            <p style="margin-bottom: 0.5rem;">Probability of being benign: {prob_benign:.3f}</p>
            <p style="margin-bottom: 0.5rem;">Probability of being malicious: {prob_malicious:.3f}</p>
        </div>
    """, unsafe_allow_html=True)

    # Add notes field and save button
    notes = st.text_area("Add notes (optional)")
    if st.button("Save Prediction"):
        db = Database()
        success, message = db.save_prediction(
            user_id=st.session_state.user_id,
            prediction=prediction_type,
            confidence_benign=prob_benign,
            confidence_malicious=prob_malicious,
            input_data=input_data,
            notes=notes
        )
        if success:
            st.success("Prediction saved successfully!")
        else:
            st.error(f"Failed to save prediction: {message}")
    
    st.markdown("""
        <p style="color: #2a5298; font-size: 0.9rem; font-style: italic;">
            This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.
        </p>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'predictor'  # Default view

def login_form():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        # Updated title with gradient effect
        st.markdown('<h1 class="form-title">CancerSense AI</h1>', unsafe_allow_html=True)
        
        # Username input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Username</label>', unsafe_allow_html=True)
        username = st.text_input("", key="login_username", placeholder="Enter your username")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Password</label>', unsafe_allow_html=True)
        password = st.text_input("", key="login_password", type="password", placeholder="Enter your password")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Login button
        if st.button("Login", key="login_button"):
            if username and password:
                try:
                    db = Database()
                    success, message = db.login_user(username, password)
                    if success:
                        # Get user_id after successful login
                        user_id = db.get_user_id(username)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id  # Set the user_id
                        st.session_state.page = 'main'
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please fill in all fields")
        
        # Divider
        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        
        # Sign up button
        if st.button("Create New Account", key="signup_redirect"):
            st.session_state.page = 'signup'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def signup_form():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        # Medical logo URL
        
        st.markdown('<h1 class="form-title">Create Account</h1>', unsafe_allow_html=True)
        
        # Username input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Username</label>', unsafe_allow_html=True)
        username = st.text_input("", key="signup_username", placeholder="Choose a username")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Email input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Email</label>', unsafe_allow_html=True)
        email = st.text_input("", key="signup_email", placeholder="Enter your email")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Password</label>', unsafe_allow_html=True)
        password = st.text_input("", key="signup_password", placeholder="Create a password")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confirm password input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<label>Confirm Password</label>', unsafe_allow_html=True)
        confirm_password = st.text_input("", key="signup_confirm_password", placeholder="Confirm your password")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sign up button
        if st.button("Sign Up", key="signup_button"):
            if username and email and password and confirm_password:
                if password != confirm_password:
                    st.error("Passwords do not match")
                    return
                
                db = Database()
                success, message = db.register_user(username, email, password)
                if success:
                    st.success(message)
                    st.info("Please login to continue")
                    st.session_state.page = 'login'
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")
        
        # Divider
        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        
        # Back to login button
        if st.button("Back to Login", key="login_redirect"):
            st.session_state.page = 'login'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def logout():
    """Function to handle logout and reset session state"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = 'login'

def show_navigation():
    # Ensure current_view is initialized
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'predictor'
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Predictor", use_container_width=True):
            st.session_state.current_view = 'predictor'
            st.rerun()
    with col2:
        if st.button("Dashboard", use_container_width=True):
            st.session_state.current_view = 'dashboard'
            st.rerun()
    with col3:
        if st.button("History", use_container_width=True):
            st.session_state.current_view = 'history'
            st.rerun()

def generate_report(history_data, username):
    """Generate a detailed medical report from prediction history"""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'CancerSense AI - Medical Analysis Report', 0, 1, 'C')
            self.set_font('Arial', '', 10)
            self.cell(0, 5, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
            self.line(10, 30, 200, 30)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'This report is generated by CancerSense AI and should be reviewed by a medical professional.', 0, 0, 'C')
            self.set_y(-10)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 6, title, 0, 1, 'L', 1)
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln()

    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Report Summary
    pdf.chapter_title('Report Summary')
    pdf.chapter_body(f"""
    Healthcare Provider: {username}
    Total Predictions: {len(history_data)}
    Period: {min([r[6] for r in history_data]).strftime('%Y-%m-%d')} to {max([r[6] for r in history_data]).strftime('%Y-%m-%d')}
    """)
    
    # Statistics
    malignant_count = sum(1 for r in history_data if r[1] == 'Malignant')
    benign_count = sum(1 for r in history_data if r[1] == 'Benign')
    
    pdf.chapter_title('Analysis Statistics')
    pdf.chapter_body(f"""
    Total Malignant Predictions: {malignant_count}
    Total Benign Predictions: {benign_count}
    Malignancy Rate: {(malignant_count/len(history_data)*100):.1f}%
    """)
    
    # Detailed Predictions
    pdf.add_page()
    pdf.chapter_title('Detailed Prediction History')
    
    for record in history_data:
        id, prediction, conf_benign, conf_malicious, input_data_str, notes, timestamp = record
        
        # Box for each prediction
        pdf.set_draw_color(100, 100, 100)
        pdf.rect(10, pdf.get_y(), 190, 60)
        
        # Prediction header
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f'Analysis #{id} - {timestamp.strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L', 1)
        
        # Main prediction info
        pdf.set_font('Arial', '', 10)
        prediction_color = (255, 0, 0) if prediction == 'Malignant' else (0, 100, 0)
        pdf.set_text_color(*prediction_color)
        pdf.cell(0, 6, f'Diagnosis: {prediction}', 0, 1)
        pdf.set_text_color(0, 0, 0)
        
        # Confidence scores
        pdf.cell(0, 6, f'Confidence Scores:', 0, 1)
        pdf.cell(90, 6, f'Benign: {conf_benign:.1%}', 0, 0)
        pdf.cell(90, 6, f'Malignant: {conf_malicious:.1%}', 0, 1)
        
        # Notes
        if notes:
            pdf.cell(0, 6, f'Clinical Notes: {notes}', 0, 1)
        
        # Key measurements
        try:
            input_data = json.loads(input_data_str)
            pdf.cell(0, 6, 'Key Measurements:', 0, 1)
            key_metrics = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 
                'area_mean', 'smoothness_mean'
            ]
            for metric in key_metrics:
                if metric in input_data:
                    # Format the metric name to be more readable
                    metric_name = metric.replace('_', ' ').title()
                    pdf.cell(0, 4, f'{metric_name}: {input_data[metric]:.3f}', 0, 1)
            pdf.ln(2)  # Add a small space after measurements
        except:
            pass
        
        pdf.ln(15)
        
        # Check if we need a new page
        if pdf.get_y() > 250:
            pdf.add_page()
    
    # Disclaimer page
    pdf.add_page()
    pdf.chapter_title('Important Notice')
    pdf.chapter_body("""
    This report is generated by CancerSense AI, an artificial intelligence-based diagnostic support tool. The predictions and analyses contained in this report should be used as supporting information only and not as a sole basis for diagnosis.

    Key Points:
    1. All predictions should be verified by qualified medical professionals
    2. This tool is designed to assist, not replace, professional medical judgment
    3. Additional clinical correlation and testing may be necessary
    4. Patient history and other clinical factors should be considered
    
    For medical professionals use only.
    """)
    
    return pdf.output(dest='S').encode('latin-1')

def generate_single_report(record, username):
    """Generate a report for a single prediction"""
    id, prediction, conf_benign, conf_malicious, input_data_str, notes, timestamp = record
    
    class PDF(FPDF):
        def header(self):
            # Center aligned title
            self.set_font('Arial', 'B', 14)
            self.cell(0, 8, 'CancerSense AI - Medical Analysis Report', 0, 1, 'C')
            # Center aligned timestamp
            self.set_font('Arial', '', 8)
            self.cell(0, 4, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
            # Horizontal line
            self.line(10, 20, 200, 20)
            self.ln(5)

        def footer(self):
            self.set_y(-12)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, 'This report is generated by CancerSense AI and should be reviewed by a medical professional.', 0, 1, 'C')

    # Create PDF object
    pdf = PDF()
    pdf.add_page()
    pdf.set_left_margin(15)  # Add left margin for better alignment
    
    # Analysis Details - Left aligned
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, 'Analysis Details', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f'ID: #{id} | Date: {timestamp.strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
    pdf.cell(0, 5, f'Provider: {username}', 0, 1, 'L')
    pdf.ln(2)
    
    # Diagnosis Result - Left aligned with color
    pdf.set_font('Arial', 'B', 11)
    if prediction == 'Malignant':
        pdf.set_text_color(255, 0, 0)
    else:
        pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 6, f'Diagnosis: {prediction}', 0, 1, 'L')
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f'Confidence (Benign): {conf_benign:.1%}', 0, 1, 'L')
    pdf.cell(0, 5, f'Confidence (Malignant): {conf_malicious:.1%}', 0, 1, 'L')
    pdf.ln(2)
    
    # Key Measurements - Left aligned
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, 'Key Measurements', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    try:
        input_data = json.loads(input_data_str)
        key_metrics = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
        for metric in key_metrics:
            if metric in input_data:
                metric_name = metric.replace('_', ' ').title()
                pdf.cell(0, 5, f'{metric_name}: {input_data[metric]:.3f}', 0, 1, 'L')
    except:
        pass
    pdf.ln(2)
    
    # Clinical Recommendations - Left aligned with bullet points
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, 'Clinical Recommendations', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    if prediction == 'Malignant':
        recommendations = [
            "1. Immediate Consultation: Schedule urgent oncologist consultation",
            "2. Further Testing: Tissue biopsy, mammogram/ultrasound, consider MRI",
            "3. Treatment Planning: Begin preliminary planning pending confirmation",
            "4. Support Services: Connect with cancer support services",
            "5. Follow-up: Schedule within 1 week"
        ]
    else:
        recommendations = [
            "1. Regular Monitoring: Continue routine screening",
            "2. Follow-up: Mammogram in 12 months, clinical examination every 6-12 months",
            "3. Risk Management: Maintain healthy lifestyle, regular self-examination",
            "4. Documentation: Keep records of all screenings"
        ]
    
    for rec in recommendations:
        pdf.cell(0, 5, rec, 0, 1, 'L')
    pdf.ln(2)
    
    # Patient Instructions - Left aligned with bullet points
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, 'Patient Instructions', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    instructions = [
        "1. Keep this report for medical records",
        "2. Share with primary healthcare provider",
        "3. Follow recommended follow-up schedule",
        "4. Report any new/changing symptoms immediately",
        "5. Maintain healthy lifestyle (exercise, diet, sleep, stress management)"
    ]
    
    for inst in instructions:
        pdf.cell(0, 5, inst, 0, 1, 'L')
    
    return pdf.output(dest='S').encode('latin-1')

def show_history():
    st.markdown("""
        <div class="history-container">
            <h2>Prediction History & Reports</h2>
            <p class="subtitle">View, analyze, and generate reports from your predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get user history
    db = Database()
    all_history = db.get_user_history(st.session_state.user_id)
    
    if not all_history:
        st.info("No predictions saved yet. Make some predictions to see them here!")
        return
    
    # Date filtering
    st.markdown("### Filter Predictions")
    col1, col2 = st.columns(2)
    
    # Convert datetime objects to date objects for filtering
    try:
        dates = [record[6].date() for record in all_history]  # timestamp is at index 6
        min_date = min(dates)
        max_date = max(dates)
        
        with col1:
            start_date = st.date_input(
                "From Date", 
                min_date,
            )
        with col2:
            end_date = st.date_input(
                "To Date", 
                max_date,
            )
        
        # Filter history based on selected dates
        filtered_history = [
            record for record in all_history 
            if start_date <= record[6].date() <= end_date
        ]
        
        if not filtered_history:
            st.warning("No predictions found in the selected date range.")
            return
        
        st.markdown(f"### Showing {len(filtered_history)} Predictions")
        
    except Exception as e:
        st.error(f"Error processing dates: {str(e)}")
        filtered_history = all_history  # Show all if date filtering fails
    
    # Display filtered history
    for record in filtered_history:
        id, prediction, confidence_benign, confidence_malicious, input_data_str, notes, timestamp = record
        
        with st.expander(f"Prediction {id} - {timestamp}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Prediction:** {prediction}")
                st.markdown(f"**Confidence (Benign):** {confidence_benign:.3f}")
                st.markdown(f"**Confidence (Malicious):** {confidence_malicious:.3f}")
            
            with col2:
                if notes:
                    st.markdown(f"**Clinical Notes:** {notes}")
                
                # Generate individual report button
                if st.button(f"🔄 Generate Report #{id}"):
                    report_pdf = generate_single_report(record, st.session_state.username)
                    st.download_button(
                        label=f"📥 Download Report #{id}",
                        data=report_pdf,
                        file_name=f"medical_report_{id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )

def show_dashboard():
    st.markdown("""
        <div class="dashboard-container">
            <h2>Analytics Dashboard</h2>
            <p class="subtitle">Overview of prediction statistics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get user history
    db = Database()
    history = db.get_user_history(st.session_state.user_id)
    
    if not history:
        st.info("No predictions available yet. Make some predictions to see analytics!")
        return
    
    # Summary Statistics in Cards
    col1, col2, col3 = st.columns(3)
    
    total_predictions = len(history)
    malignant_count = sum(1 for r in history if r[1] == 'Malignant')
    benign_count = total_predictions - malignant_count
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Predictions</h3>
                <p class="metric-value">{}</p>
            </div>
        """.format(total_predictions), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Malignant Cases</h3>
                <p class="metric-value" style="color: #ff4b4b;">{}</p>
            </div>
        """.format(malignant_count), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>Benign Cases</h3>
                <p class="metric-value" style="color: #00cc00;">{}</p>
            </div>
        """.format(benign_count), unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("### Recent Predictions")
    recent_predictions = history[-5:]  # Last 5 predictions
    recent_predictions.reverse()  # Show newest first
    
    for record in recent_predictions:
        id, prediction, conf_benign, conf_malicious, _, notes, timestamp = record
        
        # Color coding for prediction type
        pred_color = "#ff4b4b" if prediction == "Malignant" else "#00cc00"
        pred_text_color = "color: #ff4b4b;" if prediction == "Malignant" else "color: #00cc00;"
        
        st.markdown(
            f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="color: #666;">Prediction #{id} - {timestamp.strftime('%Y-%m-%d %H:%M')}</div>
                <div style="margin-top: 10px;">
                    <p style="margin: 5px 0;"><strong>Diagnosis:</strong> <span style="{pred_text_color}">{prediction}</span></p>
                    <p style="margin: 5px 0;"><strong>Confidence:</strong> {conf_malicious:.1%} if malignant</p>
                    {f'<p style="margin: 5px 0;"><strong>Notes:</strong> {notes}</p>' if notes else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    st.set_page_config(
        page_title="CancerSense AI",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    init_session_state()
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    if st.session_state.page == 'login':
        login_form()
    elif st.session_state.page == 'signup':
        signup_form()
    elif st.session_state.logged_in:
        # Header
        st.markdown(f"""
            <div style="background-color: #1e3d7b; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/4497/4497919.png" style="width: 40px; margin-right: 10px;">
                    <span style="color: white; font-size: 24px;">CancerSense AI</span>
                </div>
                <span style="color: white;">Welcome, {st.session_state.username}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        show_navigation()
        
        # Add logout button in sidebar with white text
        with st.sidebar:
            st.markdown(
                """
                <style>
                    div[data-testid="stButton"] button {
                        color: white !important;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.button("Logout", on_click=logout, type="primary")
            
        # Content based on current view
        if st.session_state.current_view == 'predictor':
            with st.sidebar:
                #st.markdown('<h3 style="color: #1e3d7b;">Cell Nuclei Measurements</h3>', unsafe_allow_html=True)
                input_data = add_sidebar()
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
            add_predictions(input_data)
        elif st.session_state.current_view == 'dashboard':
            show_dashboard()
        else:
            show_history()
        
        # Footer
        st.markdown("""
            <div class="footer">
                <p>© 2024 CancerSense AI. For medical professionals use only.</p>
                <p class="disclaimer">This tool is designed to assist medical professionals and should not be used as a substitute for professional medical diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
  main()