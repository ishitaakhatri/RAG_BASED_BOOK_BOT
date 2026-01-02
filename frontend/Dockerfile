FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# --- NEW SECTION STARTS ---
# Declare the build argument
ARG VITE_CLERK_PUBLISHABLE_KEY
# Set it as an environment variable for the build command
ENV VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY
# --- NEW SECTION ENDS ---

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