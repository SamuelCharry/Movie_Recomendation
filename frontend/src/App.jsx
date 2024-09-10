import React, { useState } from 'react';
import axios from 'axios';
import { Container, Form, Button, Row, Col, Card, InputGroup } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import NavBar from './Components/Navbar';

function App() {
    const [movieName, setMovieName] = useState('');
    const [recommendations, setRecommendations] = useState([]);

    const fetchRecommendations = async () => {
        try {
            const encodedMovieName = encodeURIComponent(movieName);
            const response = await axios.get(`http://127.0.0.1:3050/api/movies/${encodedMovieName}`);
            setRecommendations(response.data);
        } catch (error) {
            console.error('Error fetching recommendations', error);
        }
    };
   

    return (
        <>
            <NavBar/>
            <Container className="d-flex flex-column align-items-center" >
                <h1 className="my-4" style={{ fontFamily: 'Poppins, sans-serif', fontWeight:'600' }}>Movie Recommender</h1>
                <p className="my-2">
                    (<span style={{ color: 'red' }}> <strong>MERN Stack, Django</strong> </span> based Movie recommendation app)
                </p>
                <p>
                    Type Movie name to get recommendations. e.g: 
                    <span style={{ color: 'blue' }}> Avatar</span>, 
                    <span style={{ color: 'blue' }}> Aliens vs Predator: Requiem</span>, 
                    <span style={{ color: 'blue' }}> Spider-Man</span>,
                    <span style={{ color: 'blue' }}> The Avengers</span> etc.
                </p>
                <Form className="w-50">
                    <Form.Group controlId="movieName">
                        <InputGroup>
                            <Form.Control
                                type="text"
                                placeholder="Enter movie name"
                                value={movieName}
                                onChange={(e) => setMovieName(e.target.value)}
                            />
                            <Button
                                variant="primary"
                                onClick={fetchRecommendations}
                            >
                                Get Recommendations
                            </Button>
                        </InputGroup>
                    </Form.Group>
                </Form>
                <Container className="mt-4">
                    <Row>
                        {recommendations.map((movie) => (
                            <Col md={4} key={movie._id} className="mb-4">
                                <Card>
                                    <Card.Body>
                                        <Card.Title>{movie.title}</Card.Title>
                                        {movie.tagline ? (
                                            <Card.Subtitle className="mb-2 text-muted">Tagline: {movie.tagline}</Card.Subtitle>
                                        ) : (
                                            <Card.Subtitle className="mb-2" style={{ color: 'red' }}>No Tagline available</Card.Subtitle>
                                        )}
                                        <Card.Text>
                                            <strong>Release Date:</strong> {movie.release_date.slice(0, 10)}
                                        </Card.Text>
                                        <Card.Text>
                                            <strong>Duration:</strong> {movie.runtime} Minutes
                                        </Card.Text>
                                    </Card.Body>
                                </Card>
                            </Col>
                        ))}
                    </Row>
                </Container>
                <h5 style={{ color: 'red' }}>Note: Poster not available due to TMDB access issues.</h5>
                <Link to='/overview'>
                    <Button variant="link" style={{ marginTop: '20px' }}>
                        Know more about the project
                    </Button>
                </Link>
            </Container>
        </>
    );
}

export default App;
