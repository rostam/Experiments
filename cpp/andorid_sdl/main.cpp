#include "SDL.h"

// When compiling with SDL2 on some platforms (including Android), you may need to include SDL_main.h.
// #include "SDL_main.h"

int main(int argc, char* argv[]) {
    // Initialize SDL's video subsystem
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_Log("SDL_Init Error: %s", SDL_GetError());
        return 1;
    }

    // Create a window (on Android this window will be full-screen by default)
    SDL_Window* window = SDL_CreateWindow("Simple 2D Graphics",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          640, 480,
                                          SDL_WINDOW_SHOWN);
    if (!window) {
        SDL_Log("SDL_CreateWindow Error: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Create a renderer for drawing
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1,
                                                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        SDL_Log("SDL_CreateRenderer Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Main loop flag
    bool running = true;
    SDL_Event event;

    // Main loop
    while (running) {
        // Process events (for example, closing the window or touch input)
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            // For touch input on Android, you might also check for SDL_FINGERDOWN events.
        }

        // Clear the screen with a black color
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Draw a white rectangle
        SDL_Rect rect = { 200, 150, 240, 180 };
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(renderer, &rect);

        // Update the screen with any rendering performed since the previous call
        SDL_RenderPresent(renderer);
    }

    // Clean up resources before exiting
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
