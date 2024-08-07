import React from 'react';
import { Container, Card, Button, ListGroup } from 'react-bootstrap';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCopy } from '@fortawesome/free-solid-svg-icons';
import NavBar from './Navbar';

const commands = [
  {
    description: 'Clone the repository',
    command: 'git clone https://github.com/dipanjanpathak/MERN_Movie_Recomendation.git\ncd MERN_Movie_Recomendation'
  },
  {
    description: 'Install backend dependencies',
    command: 'cd backend\nnpm install'
  },
  {
    description: 'Install frontend dependencies',
    command: 'cd frontend\nnpm install'
  },
  {
    description: 'Install Python dependencies',
    command: 'cd Movie_Python\npip install -r requirements.txt'
  },
  {
    description: 'Start MongoDB',
    command: 'mongod'
  },
  {
    description: 'Start the backend server',
    command: 'cd backend\nnpm start'
  },
  {
    description: 'Start the frontend server',
    command: 'cd frontend\nnpm start'
  },
  {
    description: 'Start the Django server',
    command: 'cd ml\npython manage.py runserver'
  }
];

const ProjectDetails = () => {
  return (
    <>
      <NavBar />
      <Container className="mt-4">
        <h1 style={{ color: '#007bff', marginBottom: '1rem', textAlign: 'center' }}>Movie Recommendation System</h1>
        <p style={{ marginBottom: '2rem', textAlign: 'center' }}>
          (This project is a simple movie recommendation system that leverages various technologies to provide users with personalized movie recommendations. The system is built using a combination of <span style={{ color: 'red' }}> <strong>MongoDB, Express, Node.js, React, and Django</strong> </span>, incorporating several machine learning techniques and libraries.)
        </p>

        <h2 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Technologies Used</h2>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Backend</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><strong>MongoDB</strong>: A NoSQL database used to store movie data efficiently.</ListGroup.Item>
              <ListGroup.Item><strong>Express</strong>: A web application framework for Node.js, used to build the API endpoints.</ListGroup.Item>
              <ListGroup.Item><strong>Node.js</strong>: A JavaScript runtime used to run the backend server.</ListGroup.Item>
              <ListGroup.Item><strong>Django</strong>: A high-level Python web framework used to implement the machine learning model.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Frontend</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><strong>React</strong>: A JavaScript library for building user interfaces.</ListGroup.Item>
              <ListGroup.Item><strong>React-Bootstrap</strong>: A popular front-end framework for developing responsive and mobile-first websites.</ListGroup.Item>
              <ListGroup.Item><strong>React Router</strong>: A standard library for routing in React applications.</ListGroup.Item>
              <ListGroup.Item><strong>Font Awesome</strong>: A toolkit for icons and social logos.</ListGroup.Item>
              <ListGroup.Item><strong>Axios</strong>: A promise-based HTTP client for the browser and Node.js, used to make HTTP requests from the frontend to the backend.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Machine Learning</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><strong>Pandas</strong>: A powerful data manipulation and analysis library for Python.</ListGroup.Item>
              <ListGroup.Item><strong>NumPy</strong>: A library for numerical operations in Python.</ListGroup.Item>
              <ListGroup.Item><strong>NLTK (Natural Language Toolkit)</strong>: A suite of libraries and programs for symbolic and statistical natural language processing.</ListGroup.Item>
              <ListGroup.Item><strong>Scikit-learn</strong>: A machine learning library for Python, used for implementing the recommendation algorithm.</ListGroup.Item>
              <ListGroup.Item><strong>PorterStemmer</strong>: An algorithm for removing commoner morphological and inflexional endings from words in English.</ListGroup.Item>
              <ListGroup.Item><strong>CountVectorizer</strong>: A tool from Scikit-learn used to convert a collection of text documents to a matrix of token counts.</ListGroup.Item>
              <ListGroup.Item><strong>Cosine Similarity</strong>: A metric used to measure how similar two data points are irrespective of their size.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <h2 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Project Structure</h2>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Backend</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><code>server.js</code>: The main entry point of the backend application.</ListGroup.Item>
              <ListGroup.Item><code>models/Movie.js</code>: The Mongoose schema for the Movie collection.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Frontend</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><code>src/index.js</code>: The main entry point of the React application.</ListGroup.Item>
              <ListGroup.Item><code>src/App.js</code>: The main component that includes routing and layout.</ListGroup.Item>
              <ListGroup.Item><code>src/components/Navbar.js</code>: A navigation bar component.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <Card.Title style={{ textAlign: 'center' }}>Machine Learning</Card.Title>
            <ListGroup variant="flush">
              <ListGroup.Item><code>ml/recommendation.py</code>: Python script containing the recommendation logic using Pandas, NumPy, and Scikit-learn.</ListGroup.Item>
              <ListGroup.Item><code>ml/config.py</code>: Configuration file containing movie data and similarity matrices.</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <h2 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>How It Works</h2>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <ol style={{ textAlign: 'left', paddingLeft: '1rem' }}>
              <li><strong>Data Preparation</strong>: Movie data is stored in MongoDB. The data includes movie titles, genres, descriptions, and other relevant information.</li>
              <li><strong>Machine Learning Model</strong>: A content-based filtering model is implemented using Scikit-learn. The model uses CountVectorizer to convert movie descriptions into a matrix of token counts, and Cosine Similarity to compute the similarity between movies.</li>
              <li><strong>API Endpoints</strong>: The Express server exposes API endpoints to fetch movie data and recommendations. These endpoints interact with the MongoDB database and the Django machine learning model.</li>
              <li><strong>Frontend</strong>: The React application provides a user-friendly interface for interacting with the recommendation system. Users can search for movies and get personalized recommendations.</li>
            </ol>
          </Card.Body>
        </Card>

        <h2 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Getting Started</h2>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <h3 style={{ textAlign: 'center' }}>Prerequisites</h3>
            <ListGroup variant="flush">
              <ListGroup.Item>Node.js</ListGroup.Item>
              <ListGroup.Item>Python</ListGroup.Item>
              <ListGroup.Item>MongoDB</ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>

        <h3 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Installation</h3>
        {commands.slice(0, 4).map((cmd, index) => (
          <Card key={index} className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
            <Card.Body style={{ backgroundColor: '#000', color: '#fff', padding: '1rem', borderRadius: '5px' }}>
              <Card.Title>{cmd.description}</Card.Title>
              <pre style={{ textAlign: 'left', paddingLeft: '1rem' }}>{cmd.command}</pre>
              <CopyToClipboard text={cmd.command}>
                <Button variant="primary" size="sm">
                  <FontAwesomeIcon icon={faCopy} /> Copy
                </Button>
              </CopyToClipboard>
            </Card.Body>
          </Card>
        ))}

        <h3 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Running the Application</h3>
        {commands.slice(4).map((cmd, index) => (
          <Card key={index + 5} className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
            <Card.Body style={{ backgroundColor: '#000', color: '#fff', padding: '1rem', borderRadius: '5px' }}>
              <Card.Title>{cmd.description}</Card.Title>
              <pre style={{ textAlign: 'left', paddingLeft: '1rem' }}>{cmd.command}</pre>
              <CopyToClipboard text={cmd.command}>
                <Button variant="primary" size="sm">
                  <FontAwesomeIcon icon={faCopy} /> Copy
                </Button>
              </CopyToClipboard>
            </Card.Body>
          </Card>
        ))}

        <h3 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>Usage</h3>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <p style={{ textAlign: 'left' }}>Navigate to <code>http://localhost:3000</code> in your browser to use the application. Use the search bar to find movies and get recommendations based on the selected movie.</p>
          </Card.Body>
        </Card>

        <h2 style={{ color: '#dc3545', marginBottom: '1rem', textAlign: 'center' }}>License</h2>
        <Card className="mb-4" style={{ margin: '0 auto', maxWidth: '800px' }}>
          <Card.Body>
            <p style={{ textAlign: 'left' }}>This project is licensed under the MIT License.</p>
          </Card.Body>
        </Card>
      </Container>
    </>
  );
};

export default ProjectDetails;
