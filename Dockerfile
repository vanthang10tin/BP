# ========================================================
# Stage 1: Build Blazor App (.NET 10)
# ========================================================
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS blazor-build
WORKDIR /src

# Copy project file and restore dependencies
COPY frontend/BlazorApp/BlazorApp.csproj ./frontend/BlazorApp/
RUN dotnet restore frontend/BlazorApp/BlazorApp.csproj

# Copy all Blazor source files and publish release
COPY frontend/BlazorApp/ ./frontend/BlazorApp/
RUN dotnet publish frontend/BlazorApp/BlazorApp.csproj -c Release -o /app/blazor && \
    test -f /app/blazor/wwwroot/_framework/blazor.web.js || (echo "ERROR: blazor.web.js is missing from published output!" && exit 1)

# ========================================================
# Stage 2: Final Production Runtime (ASP.NET 10 + Python 3)
# ========================================================
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app

# Install Python 3, pip, curl, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy backend, demo data, and published Blazor app
COPY backend/ ./backend/
COPY demo_data/ ./demo_data/
COPY DEMO_PACK/ ./DEMO_PACK/
COPY --from=blazor-build /app/blazor ./blazor/

# Environment settings
ENV PORT=5000
ENV QC_PUBLIC_DEMO=true
ENV QcApi__BaseUrl=http://127.0.0.1:8000
ENV ASPNETCORE_ENVIRONMENT=Production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Wait for API readiness and exit if either child dies, so Render can restart it.
COPY start_container.py ./
ENTRYPOINT ["python3", "-u", "/app/start_container.py"]
