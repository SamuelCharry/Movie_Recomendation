
# 🎬 MERN Stack Movie Recommendation System

![Workflow Diagram](https://github.com/dipanjanpathak/MERN_Movie_Recomendation/blob/master/frontend/src/images/movie-workflow-diagram.png)

This project is a simple movie recommendation system that leverages various technologies to provide users with personalized movie recommendations. The system is built using a combination of MongoDB, Express, Node.js, React, and Django, incorporating several machine learning techniques and libraries.

## Table of Contents
1. [Technologies Used](#technologies-used)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Workflow](#workflow)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Application](#running-the-application)
6. [Usage](#usage)
7. [License](#license)

## Technologies Used

### Backend
- **MongoDB**: A NoSQL database used to store movie data efficiently.
- **Express**: A web application framework for Node.js, used to build the API endpoints.
- **Node.js**: A JavaScript runtime used to run the backend server.
- **Django**: A high-level Python web framework used to implement the machine learning model.

### Frontend
- **React**: A JavaScript library for building user interfaces.
- **React-Bootstrap**: A popular front-end framework for developing responsive and mobile-first websites.
- **React Router**: A standard library for routing in React applications.
- **Font Awesome**: A toolkit for icons and social logos.
- **Axios**: A promise-based HTTP client for the browser and Node.js, used to make HTTP requests from the frontend to the backend.

### Machine Learning
- **Pandas**: A powerful data manipulation and analysis library for Python.
- **NumPy**: A library for numerical operations in Python.
- **NLTK (Natural Language Toolkit)**: A suite of libraries and programs for symbolic and statistical natural language processing.
- **Scikit-learn**: A machine learning library for Python, used for implementing the recommendation algorithm.
- **PorterStemmer**: An algorithm for removing common morphological and inflexional endings from words in English.
- **CountVectorizer**: A tool from Scikit-learn used to convert a collection of text documents to a matrix of token counts.
- **Cosine Similarity**: A metric used to measure how similar two data points are irrespective of their size.

## Project Structure

### Backend
- **server.js**: The main entry point of the backend application.
- **models/Movie.js**: The Mongoose schema for the Movie collection.

### Frontend
- **src/index.js**: The main entry point of the React application.
- **src/App.js**: The main component that includes routing and layout.
- **src/components/Navbar.js**: A navigation bar component.

### Machine Learning
- **ml/recommendation.py**: Python script containing the recommendation logic using Pandas, NumPy, and Scikit-learn.
- **ml/config.py**: Configuration file containing movie data and similarity matrices.

## How It Works

### Data Preparation
Movie data is stored in MongoDB. The data includes movie titles, genres, descriptions, and other relevant information.

### Machine Learning Model
A content-based filtering model is implemented using Scikit-learn. The model uses `CountVectorizer` to convert movie descriptions into a matrix of token counts, and `Cosine Similarity` to compute the similarity between movies.

### API Endpoints
The Express server exposes API endpoints to fetch movie data and recommendations. These endpoints interact with the MongoDB database and the Django machine learning model.

### Frontend
The React application provides a user-friendly interface for interacting with the recommendation system. Users can search for movies and get personalized recommendations.

## Workflow

Here is an overview of how the system works:

1. **User Input**:
   - The user inputs a movie name in the React frontend.
   
2. **Frontend Request**:
   - The React app sends a POST request to the **Express server**, passing the movie name.
   
3. **Movie Data Fetch**:
   - The Express server queries **MongoDB** to fetch relevant movie data.
   
4. **Machine Learning**:
   - Express sends the movie ID to **Django**, which uses a content-based filtering model to get similar movies.
   
5. **Recommendation**:
   - Django sends the recommended movie IDs back to Express, which then queries MongoDB for the details of those recommended movies.
   
6. **Display Results**:
   - Express sends the recommended movies back to React, which displays them to the user.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js**
- **Python**
- **MongoDB**

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dipanjanpathak/MERN_Movie_Recomendation.git
   cd MERN_Movie_Recomendation
   ```

2. Install backend dependencies:

   ```bash
   cd backend
   npm install
   ```

3. Install frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

4. Install Python dependencies:

   ```bash
   cd Movie_Python
   pip install -r requirements.txt
   ```

### Running the Application

1. Start MongoDB:

   ```bash
   mongod
   ```

2. Start the backend server:

   ```bash
   cd backend
   npm start
   ```

3. Start the frontend server:

   ```bash
   cd frontend
   npm start
   ```

4. Start the Django server:

   ```bash
   cd Movie_Python
   python manage.py runserver
   ```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000) in your browser to use the application. Use the search bar to find movies and get recommendations based on the selected movie.

## License

This project is licensed under the MIT License.

---

