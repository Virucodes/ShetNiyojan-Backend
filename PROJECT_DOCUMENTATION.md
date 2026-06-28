# ShetNiyojan - Intelligent Agricultural Management System

## 🌾 Project Overview

**ShetNiyojan** (meaning "Farm Planning" in Hindi) is a comprehensive intelligent agricultural management system that empowers farmers with data-driven decision-making tools for optimal crop management, disease detection, and supply chain optimization. The platform combines modern web technologies with advanced machine learning models to provide actionable insights for sustainable and profitable farming.

## 🎯 Mission Statement

To modernize agriculture through technology-driven solutions, helping farmers make informed decisions that increase productivity, reduce costs, and promote sustainable farming practices.

## 🚀 Key Features

### 1. **Smart Crop Management**
- **Crop Recommendation System**: AI-powered recommendations based on soil composition, climate data, and historical patterns
- **Yield Prediction**: Machine learning models to forecast harvest amounts and optimize planting strategies
- **Crop Health Monitoring**: Real-time monitoring of crop conditions and growth stages

### 2. **AI-Powered Disease Detection**
- **Plant Disease Analysis**: Upload plant images for instant disease identification using computer vision
- **Treatment Recommendations**: Detailed suggestions for disease treatment and prevention
- **Health Scoring**: Comprehensive health assessment of crops with actionable insights

### 3. **Supply Chain Optimization**
- **Transport Route Optimization**: Find the most profitable markets for crop sales
- **Price Analysis**: Real-time price comparison across different cities and markets
- **Cost-Benefit Analysis**: Calculate transport costs vs. profit margins for informed decisions
- **Interactive Maps**: Visual representation of optimal transport routes

### 4. **Fertilizer Recommendation System**
- **Smart Fertilizer Suggestions**: AI-powered recommendations based on soil conditions, crop type, and weather
- **Nutrient Analysis**: Detailed analysis of soil nutrients (Nitrogen, Phosphorus, Potassium)
- **Cost Optimization**: Suggest the most cost-effective fertilizer combinations

### 5. **Farm Dashboard**
- **Real-time Monitoring**: Live updates on farm activities, weather conditions, and crop status
- **Activity Logging**: Track and manage agricultural activities and tasks
- **Weather Integration**: Current weather data and forecasts for better planning
- **Yield Tracking**: Monitor current and historical crop yields

### 6. **Marketplace & Trading**
- **Crop Marketplace**: Connect farmers with buyers and traders
- **Price Discovery**: Access to current market prices and trends
- **Lease Marketplace**: Platform for agricultural land leasing and rental

### 7. **AI Chatbot Assistant**
- **24/7 Support**: Intelligent chatbot for agricultural queries and guidance
- **Expert Advice**: Access to farming best practices and recommendations
- **Multi-language Support**: Available in multiple languages for broader accessibility

## 🏗️ Technical Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript for type safety
- **Build Tool**: Vite for fast development and optimized builds
- **Styling**: Tailwind CSS with custom agricultural theme
- **UI Components**: Shadcn UI component library for consistent design
- **State Management**: React Context API for global state
- **Routing**: React Router for seamless navigation
- **Charts**: Recharts for data visualization

### Backend (Flask + Python)
- **Framework**: Flask for RESTful API development
- **Database**: MongoDB for flexible data storage
- **Authentication**: JWT-based secure authentication
- **File Upload**: Secure image upload for disease detection
- **CORS**: Cross-origin resource sharing for frontend integration

### Machine Learning & AI
- **Plant Disease Detection**: 
  - Model: MobileNetV2 trained on plant disease dataset
  - Framework: PyTorch with Transformers library
  - Accuracy: High-precision disease identification
- **Crop Recommendation**: 
  - Algorithm: Ensemble methods with AdaBoost
  - Features: Soil composition, weather, location data
- **Fertilizer Recommendation**:
  - Model: XGBoost for optimal fertilizer suggestions
  - Features: Soil nutrients, crop type, environmental factors

### External APIs & Services
- **Groq API**: AI-powered chatbot and insights generation
- **Google Generative AI**: Advanced AI capabilities
- **Mandi API**: Real-time agricultural market data
- **Weather APIs**: Current and forecast weather data

## 📁 Project Structure

```
ShetNiyojan/
├── Frontend/                    # React TypeScript application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── dashboard/       # Dashboard-specific components
│   │   │   ├── ui/             # Base UI components (Shadcn)
│   │   │   └── common/         # Shared components
│   │   ├── pages/              # Main application pages
│   │   ├── lib/                # Utilities and configurations
│   │   └── hooks/              # Custom React hooks
│   ├── public/                 # Static assets
│   └── package.json            # Frontend dependencies
├── Backend/                     # Flask Python application
│   ├── models/                 # Machine learning models
│   │   ├── crop_recommendation/
│   │   └── *.pkl              # Trained model files
│   ├── datasets/               # Training datasets
│   ├── scripts/                # Data processing scripts
│   ├── uploads/                # File upload directory
│   ├── app.py                  # Main Flask application
│   ├── db.py                   # Database configuration
│   └── requirements.txt        # Python dependencies
└── README.md                   # Project overview
```

## 🛠️ Installation & Setup

### Prerequisites
- **Node.js** 16+ and npm
- **Python** 3.8+
- **MongoDB** (local or cloud instance)
- **Groq API Key** (for AI features)

### Frontend Setup
```bash
# Navigate to frontend directory
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd Backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# GROQ_API_KEY=your_groq_api_key
# MONGODB_URI=your_mongodb_connection_string

# Start Flask server
python app.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the Backend directory:
```env
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=mongodb://localhost:27017/shetniyojan
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### API Endpoints

#### Authentication
- `POST /api/users/register` - User registration
- `POST /api/users/login` - User authentication
- `GET /api/users/profile` - Get user profile

#### Crop Management
- `POST /api/crop-recommendation` - Get crop recommendations
- `POST /api/predict-fertilizer` - Fertilizer recommendations
- `GET /api/yields` - Get yield data
- `POST /api/yields` - Add new yield data

#### Disease Detection
- `POST /api/plant-disease-analysis` - Analyze plant images for diseases

#### Supply Chain
- `POST /api/supply-chain/optimize` - Optimize transport routes
- `GET /api/market-prices` - Get current market prices

#### Chatbot
- `POST /api/chat` - Chat with AI assistant

## 🎨 User Interface

### Design System
- **Color Palette**: 
  - Primary Green: `#2C5F34` (Agricultural theme)
  - Secondary Orange: `#F97316` (Accent color)
  - Background: `#F8FAFC` (Light gray)
- **Typography**: Modern, readable fonts optimized for all devices
- **Responsive Design**: Mobile-first approach with tablet and desktop optimization

### Key Pages
1. **Landing Page**: Hero section with feature highlights
2. **Dashboard**: Centralized farm management interface
3. **Crop Prediction**: AI-powered crop recommendation tool
4. **Crop Health**: Disease detection and monitoring
5. **Supply Chain**: Transport optimization and market analysis
6. **Marketplace**: Trading and leasing platform

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt for secure password storage
- **CORS Protection**: Configured for secure cross-origin requests
- **Input Validation**: Server-side validation for all inputs
- **File Upload Security**: Secure image upload with validation

## 📊 Data Management

### Database Collections
- **Users**: User profiles and authentication data
- **Yields**: Crop yield records and historical data
- **Activities**: Farm activity logs and task management
- **Disease Reports**: Plant disease analysis results

### Machine Learning Models
- **Crop Recommendation Model**: `crop_recommendation_model.pkl`
- **Fertilizer Model**: `xgb_fertilizer_model.pkl`
- **Disease Detection**: Pre-trained MobileNetV2 model
- **Encoders**: Label encoders for categorical data

## 🌍 Deployment

### Frontend Deployment
```bash
# Build for production
npm run build

# Deploy to Vercel/Netlify
# The build folder contains optimized static files
```

### Backend Deployment
```bash
# Using Gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or deploy to cloud platforms like Heroku, AWS, or DigitalOcean
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- **IoT Integration**: Connect with soil sensors and weather stations
- **Mobile App**: Native mobile application for field use
- **Blockchain**: Supply chain transparency and traceability
- **Satellite Imagery**: Remote crop monitoring using satellite data
- **Multi-language Support**: Expand language options for global reach
- **Advanced Analytics**: Predictive analytics for market trends

## 📞 Support

For support, email support@shetniyojan.com or join our community forum.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**ShetNiyojan** - Empowering farmers with intelligent agricultural solutions for a sustainable future.

