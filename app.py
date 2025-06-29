import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_mail import Mail, Message

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'vishal.gusknp2022@gmail.com')

mail = Mail(app)

@app.route('/')
def index():
    """Main portfolio page"""
    # Portfolio data based on Vishal's resume
    portfolio_data = {
        'name': 'Vishal Maurya',
        'title': 'Data Scientist & ML Engineer',
        'email': 'vishal.gusknp2022@gmail.com',
        'phone': '+91-9260991607',
        'github': 'https://github.com/vishal-maurya',
        'linkedin': 'https://linkedin.com/in/vishal-maurya',
        'kaggle': 'https://www.kaggle.com/mauryavishal0',
        'education': {
            'degree': 'Bachelor of Technology (Hons.)',
            'field': 'Metallurgical and Materials Science Engineering',
            'institution': 'National Institute of Technology, Jamshedpur',
            'period': '2022-26',
            'cgpa': '7.32',
            'reg_id': '2022UGMM081',
            'semester_wise': [
                {'semester': 1, 'sgpa': 7.25, 'cgpa': 7.25},
                {'semester': 2, 'sgpa': 7.78, 'cgpa': 7.51},
                {'semester': 3, 'sgpa': 6.92, 'cgpa': 7.28},
                {'semester': 4, 'sgpa': 7.08, 'cgpa': 7.21},
                {'semester': 5, 'sgpa': 7.78, 'cgpa': 7.32}
            ],
            'school_education': [
                {
                    'level': 'Senior Secondary (XII)',
                    'year': '2021',
                    'board': 'CBSE',
                    'institution': 'Jawahar Navodaya Vidyalaya',
                    'percentage': '87.8%'
                },
                {
                    'level': 'Secondary School (X)',
                    'year': '2019',
                    'board': 'CBSE',
                    'institution': 'Jawahar Navodaya Vidyalaya',
                    'percentage': '93.2%'
                }
            ]
        },
        'projects': [
            {
                'title': 'Customer Segmentation + Sentiment Analysis Dashboard',
                'description': 'Built an end-to-end ML pipeline using KMeans and logistic regression on a Kaggle dataset, with an interactive Plotly Dash dashboard. Applied NLP techniques and trained models for sentiment prediction and behavioral segmentation.',
                'technologies': ['Python', 'pandas', 'scikit-learn', 'NLTK', 'TF-IDF', 'KMeans', 'Logistic Regression', 'Seaborn', 'Plotly', 'Dash'],
                'highlights': ['End-to-end ML pipeline', 'Interactive dashboard', 'Business stakeholder presentation'],
                'image': 'https://pixabay.com/get/g36208a1b77913678b03a464f36cd96f5c06b18f05f8191572d32a013c4408124a480de5b92a760d6428d9f066a0273918dfa9a993f172a5b2dd8590f6640e6fb_1280.jpg',
                'demo_url': '/demo/customer-segmentation',
                'github_url': 'https://github.com/vishal-maurya/customer-segmentation-dashboard',
                'kaggle_dataset': 'https://www.kaggle.com/mauryavishal0/datasets'
            },
            {
                'title': 'NDVI-Based Land Cover Classification',
                'description': 'Summer Analytics Hackathon 2025 - IIT Guwahati. Built a logistic regression model to classify land cover types from satellite NDVI time series data. Achieved 79.09% accuracy and ranked 474/1394 on public leaderboard.',
                'technologies': ['Python', 'pandas', 'scikit-learn', 'Logistic Regression', 'matplotlib', 'seaborn'],
                'highlights': ['79.09% accuracy', 'Ranked 474/1394', 'Advanced feature engineering', 'Satellite data analysis'],
                'image': 'https://pixabay.com/get/gf74fcb45f5ab4b95e6e74e48063559ee19ce5cd7c801639db20adacaf496cc2ab052edebff442d0643f8c1ff6c20a5811e767a121469a8a3ecaf78d362f03227_1280.jpg',
                'demo_url': '/demo/land-cover-classification',
                'github_url': 'https://github.com/vishal-maurya/ndvi-land-cover-classification',
                'kaggle_competition': 'https://www.kaggle.com/competitions/iit-guwahati-summer-analytics-2025'
            }
        ],
        'experience': [
            {
                'title': 'Data Science Trainee',
                'company': 'Consulting and Analytics Club, IIT Guwahati — Summer Analytics 2025',
                'period': 'May 2025 – June 2025 (2 months)',
                'location': 'Online',
                'achievements': [
                    'Intensive training in machine learning and analytics through real-world case studies',
                    'Built and evaluated logistic regression pipelines with scaling, imputation, and feature engineering',
                    'Learning advanced algorithms: Random Forests, SVMs, XGBoost, Clustering',
                    'Developing skills in model selection, cross-validation, hyperparameter tuning'
                ]
            },
            {
                'title': 'Data Analyst Trainee',
                'company': 'Ira Skills',
                'period': 'July 2024 – Aug 2024 (2 months)',
                'location': 'Online',
                'achievements': [
                    'Gained practical skills in MS Excel, Power BI, and PowerPoint',
                    'Analyzed historical sales data of pizza restaurant chain for KPIs',
                    'Employed Microsoft Query to automate data fetching',
                    'Built interactive Power BI dashboards for sales trends visualization',
                    'Presented actionable insights through structured PowerPoint presentations'
                ]
            }
        ],
        'skills': {
            'languages': ['Python', 'SQL'],
            'libraries': ['Pandas', 'Seaborn', 'Matplotlib', 'OpenCV', 'Scikit-learn', 'Tensorflow', 'Plotly'],
            'tools': ['Google Colab', 'Kaggle', 'Jupyter Notebook', 'VS Code'],
            'frameworks': ['Dash'],
            'databases': ['IBMdb2', 'MySQL'],
            'analytics_tools': ['Microsoft Excel', 'Power BI', 'PowerPoint', 'Tableau', 'Microsoft Power Query']
        },
        'interests': ['Data Analysis', 'Machine Learning'],
        'soft_skills': ['Problem Solving', 'Self-learning', 'Presentation', 'Adaptability'],
        'achievements': [
            {
                'title': 'State Level Scholarship Winner',
                'description': 'Won the State level Scholarship test organized by CSRL DELHI and GAIL INDIA',
                'icon': 'fas fa-trophy'
            },
            {
                'title': 'Batch Topper 2017',
                'description': 'Topper of batch 2017 (CBSE Board) of Jawahar Navodaya Vidyalaya, Barabanki',
                'icon': 'fas fa-medal'
            },
            {
                'title': 'School Rank 2nd',
                'description': 'Secured 2nd rank in school CBSE 12th Board examination',
                'icon': 'fas fa-award'
            },
            {
                'title': 'FFE Scholar',
                'description': 'Foundation for Excellence Scholar with Core Employability Skills training from Wadhwani Foundation',
                'icon': 'fas fa-graduation-cap'
            },
            {
                'title': 'GAIL Utkarsh Super 100',
                'description': 'Completed JEE preparation at GAIL Utkarsh Super 100 - joint initiative of GAIL India and CSRL',
                'icon': 'fas fa-rocket'
            },
            {
                'title': 'Subject Expert at Chegg India',
                'description': 'Physics Subject Matter Expert with 889+ questions solved',
                'icon': 'fas fa-atom'
            }
        ],
        'certifications': [
            {
                'name': 'Google Data Analytics Professional Certificate',
                'issuer': 'Google',
                'date': '2024',
                'credential_url': 'https://www.linkedin.com/in/vishal-maurya/details/certifications/',
                'skills': ['Data Analysis', 'Data Visualization', 'SQL', 'Tableau', 'R Programming']
            },
            {
                'name': 'IBM Data Science Professional Certificate',
                'issuer': 'IBM',
                'date': '2024',
                'credential_url': 'https://www.linkedin.com/in/vishal-maurya/details/certifications/',
                'skills': ['Python', 'Machine Learning', 'Data Science', 'Jupyter Notebooks', 'GitHub']
            },
            {
                'name': 'Microsoft Azure AI Fundamentals',
                'issuer': 'Microsoft',
                'date': '2024',
                'credential_url': 'https://www.linkedin.com/in/vishal-maurya/details/certifications/',
                'skills': ['AI', 'Machine Learning', 'Azure', 'Computer Vision', 'Natural Language Processing']
            }
        ]
    }
    
    return render_template('index.html', data=portfolio_data)

@app.route('/contact', methods=['POST'])
def contact():
    """Handle contact form submission"""
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('All fields are required.', 'error')
            return redirect(url_for('index') + '#contact')
        
        # Create email message
        msg = Message(
            subject=f"Portfolio Contact: {subject}",
            recipients=['vishal.gusknp2022@gmail.com'],
            body=f"""
New contact form submission:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}
            """,
            reply_to=email
        )
        
        # Send email
        mail.send(msg)
        flash('Thank you for your message! I will get back to you soon.', 'success')
        
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}")
        flash('Sorry, there was an error sending your message. Please try again later.', 'error')
    
    return redirect(url_for('index') + '#contact')

@app.route('/demo/customer-segmentation')
def demo_customer_segmentation():
    """Demo page for Customer Segmentation project"""
    return render_template('demos/customer_segmentation.html')

@app.route('/demo/land-cover-classification')
def demo_land_cover_classification():
    """Demo page for NDVI Land Cover Classification project"""
    return render_template('demos/land_cover_classification.html')

@app.route('/api/demo/customer-segmentation', methods=['POST'])
def api_customer_segmentation():
    """API endpoint for customer segmentation demo"""
    import json
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    try:
        # Sample customer data for demo
        np.random.seed(42)
        n_customers = 100
        
        # Generate synthetic customer data
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(35, 10, n_customers),
            'annual_spending': np.random.normal(5000, 2000, n_customers),
            'frequency_score': np.random.normal(50, 15, n_customers),
            'recency_days': np.random.exponential(30, n_customers)
        }
        
        df = pd.DataFrame(customer_data)
        df = df[(df['age'] > 18) & (df['age'] < 80)]
        df = df[df['annual_spending'] > 0]
        
        # Prepare features for clustering
        features = ['age', 'annual_spending', 'frequency_score', 'recency_days']
        X = df[features]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)
        
        # Define segment labels
        segment_labels = {
            0: 'Loyal Customers',
            1: 'High-Value Customers', 
            2: 'At-Risk Customers',
            3: 'New Customers'
        }
        
        df['segment_label'] = df['segment'].map(segment_labels)
        
        # Calculate segment statistics
        segment_stats = df.groupby('segment_label').agg({
            'age': 'mean',
            'annual_spending': 'mean',
            'frequency_score': 'mean',
            'recency_days': 'mean'
        }).round(2)
        
        result = {
            'customer_data': df.to_dict('records'),
            'segment_stats': segment_stats.to_dict('index'),
            'total_customers': len(df),
            'segments_count': df['segment_label'].value_counts().to_dict()
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)}), 500

@app.route('/api/demo/land-cover-classification', methods=['POST'])
def api_land_cover_classification():
    """API endpoint for land cover classification demo"""
    import json
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    try:
        # Sample NDVI data for demo
        np.random.seed(42)
        n_samples = 500
        
        # Generate synthetic NDVI time series data
        land_cover_types = ['Forest', 'Farmland', 'Grassland', 'Urban', 'Water']
        
        # Different NDVI patterns for different land covers
        ndvi_data = []
        labels = []
        
        for i in range(n_samples):
            land_type = np.random.choice(land_cover_types)
            
            if land_type == 'Forest':
                base_ndvi = np.random.normal(0.8, 0.1, 12)  # High NDVI
            elif land_type == 'Farmland':
                base_ndvi = np.random.normal(0.6, 0.15, 12)  # Seasonal variation
            elif land_type == 'Grassland':
                base_ndvi = np.random.normal(0.4, 0.1, 12)  # Medium NDVI
            elif land_type == 'Urban':
                base_ndvi = np.random.normal(0.2, 0.05, 12)  # Low NDVI
            else:  # Water
                base_ndvi = np.random.normal(0.1, 0.02, 12)  # Very low NDVI
            
            # Clip NDVI values to valid range
            base_ndvi = np.clip(base_ndvi, -1, 1)
            
            # Calculate features
            features = {
                'mean_ndvi': np.mean(base_ndvi),
                'std_ndvi': np.std(base_ndvi),
                'max_ndvi': np.max(base_ndvi),
                'min_ndvi': np.min(base_ndvi),
                'range_ndvi': np.max(base_ndvi) - np.min(base_ndvi),
                'trend': np.polyfit(range(12), base_ndvi, 1)[0]  # Linear trend
            }
            
            ndvi_data.append(features)
            labels.append(land_type)
        
        # Create DataFrame
        df = pd.DataFrame(ndvi_data)
        df['land_cover'] = labels
        
        # Prepare features and target
        feature_cols = ['mean_ndvi', 'std_ndvi', 'max_ndvi', 'min_ndvi', 'range_ndvi', 'trend']
        X = df[feature_cols]
        y = df['land_cover']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance (coefficients)
        feature_importance = {}
        for i, feature in enumerate(feature_cols):
            feature_importance[feature] = np.mean(np.abs(model.coef_[:, i]))
        
        result = {
            'accuracy': round(accuracy * 100, 2),
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'land_cover_distribution': df['land_cover'].value_counts().to_dict(),
            'feature_importance': feature_importance,
            'sample_predictions': {
                'actual': y_test.head(10).tolist(),
                'predicted': y_pred[:10].tolist()
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.getenv("PORT", 3000)), debug=False)
