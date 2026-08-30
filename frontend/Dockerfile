FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# --- BUILD ARGS ---
# Declare the build arguments
ARG VITE_CLERK_PUBLISHABLE_KEY
ARG VITE_API_URL
# Set them as environment variables for the build command
ENV VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY
ENV VITE_API_URL=$VITE_API_URL
# --- END BUILD ARGS ---

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
# Copy the build output to Nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html
# We will use a custom Nginx config in the main compose file, 
# so we don't need a complex config here for now.
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]