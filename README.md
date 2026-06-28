# ShetNiyojan

ShetNiyojan is an intelligent agricultural management system that helps farmers make data-driven decisions for optimal crop management and disease detection.

## Project Overview

ShetNiyojan combines modern frontend technologies with machine learning powered backend to provide farmers with valuable insights for agricultural planning and management.

## Project Structure

The project is divided into two main parts:

### Frontend
- Built with React + TypeScript + Vite
- Uses Tailwind CSS and Shadcn UI components
- Responsive design for all devices
- Located in the `Frontend` directory

### Backend
- Built with Flask and Python
- MongoDB database integration
- Machine learning models for crop recommendation and disease detection
- Groq Vision API integration for plant disease analysis
- Located in the `Backend` directory

## Features

- **User Authentication**: Secure login and registration system
- **Dashboard**: Centralized view of agricultural metrics and insights
- **Crop Health Monitoring**: 
  - Upload plant images for disease detection
  - AI-powered analysis of plant diseases
  - Detailed recommendations for treatment
- **Crop Recommendation**:
  - Soil composition analysis
  - Climate-based crop suggestions
  - Detailed growing recommendations
- **Yield Prediction**: Forecast potential harvest amounts based on historical data
- **Task Management**: Schedule and track agricultural activities

## Getting Started

### Frontend Setup
1. Navigate to the Frontend directory:
   ```bash
   cd Frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

### Backend Setup
1. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (see `.env.example` for required variables)

5. Start the backend server:
   ```bash
   python app.py
   ```

## API Endpoints

### Authentication
- `POST /api/users/register`: Register a new user
- `POST /api/users/login`: Authenticate user and get token
- `GET /api/users/profile`: Get current user profile

### Plant Disease Analysis
- `POST /api/plant-disease-analysis`: Upload plant image for disease detection

### Crop Recommendation
- `POST /api/crop-recommendation`: Get crop recommendations based on soil and climate data

## Technologies Used

### Frontend
- React 18 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Shadcn UI component library
- Axios for API communication

### Backend
- Flask for API development
- MongoDB for database
- Python for backend logic
- Machine Learning models for predictions
- Groq API for AI image analysis

## Requirements

- Node.js 16+
- Python 3.8+
- MongoDB
- Groq API key for plant disease detection

## License

This project is licensed under the MIT License. 