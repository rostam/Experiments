#include <SFML/Graphics.hpp>

const float WINDOW_WIDTH = 800;
const float WINDOW_HEIGHT = 600;
const float GRAVITY = 0.5f;  // Gravity acceleration
const float BOUNCE_FACTOR = 0.7f; // Energy loss on bounce
const float BALL_RADIUS = 20.0f;
const float GROUND_Y = WINDOW_HEIGHT - BALL_RADIUS * 2;

int main() {
    // Create the window
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Bouncing Ball with Gravity");
    window.setFramerateLimit(60); // Set frame rate limit

    // Ball properties
    sf::CircleShape ball(BALL_RADIUS);
    ball.setFillColor(sf::Color::Red);
    ball.setPosition(WINDOW_WIDTH / 2, 50); // Initial position

    // Physics properties
    float velocityY = 0.0f; // Initial velocity

    // Game loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Apply gravity
        velocityY += GRAVITY;
        ball.move(0, velocityY);

        // Collision with ground
        if (ball.getPosition().y >= GROUND_Y) {
            ball.setPosition(ball.getPosition().x, GROUND_Y);
            velocityY = -velocityY * BOUNCE_FACTOR; // Reverse and reduce speed on bounce
        }

        // Clear, draw, and display
        window.clear(sf::Color::Black);
        window.draw(ball);
        window.display();
    }

    return 0;
}
