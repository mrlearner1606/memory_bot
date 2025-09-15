# Memory Bot - Personal Knowledge Management System

A web-based application that stores and retrieves personal memories using AI-powered natural language processing. The bot automatically classifies whether you're adding new information or asking questions, then responds appropriately.

## Features

• **Smart Memory Storage** - Save personal information in natural language
• **Intelligent Retrieval** - Ask questions and get answers from your stored data
• **Auto Classification** - Automatically detects if you're storing info or asking questions
• **API Failover System** - Uses multiple OpenRouter API keys for reliability
• **Secure Web Interface** - Password-protected access with clean UI
• **Airtable Integration** - Reliable cloud storage for your memories

## How It Works

### Storage Mode (INSERT)
When you input: "I graduated from college on May 15, 2020"
- **Knowledge**: "I graduated from college on May 15, 2020"
- **Reference**: "graduation, college, education"
- **Date**: "2020-05-15"

### Query Mode (SEARCH)
When you ask: "When did I graduate?"
- Searches stored records for relevant keywords
- Returns natural language answer based on your data

## Prerequisites

Before installation, ensure you have:

• Python 3.8 or higher
• Airtable account (free tier available)
• 3 OpenRouter accounts (free tier available)
• Basic command line knowledge

## Installation & Setup

### 1. Clone or Download the Code

Create a new folder and save the `app.py` file inside it.

### 2. Install Dependencies

```bash
pip install flask python-dotenv requests
```

### 3. Airtable Configuration

**Step 1: Create Database**
1. Sign up at [airtable.com](https://airtable.com)
2. Create new base called "Memory"
3. Create table with these exact column names:

| Column Name | Field Type | Purpose |
|-------------|------------|---------|
| Knowledge | Long text | Stores the actual memory/information |
| Reference | Single line text | Keywords for searching (comma-separated) |
| Date | Date | When the memory was recorded |

**Step 2: Get API Credentials**
1. Go to Account → Developer Hub
2. Create new personal access token
3. Select your base and grant full permissions
4. Copy the token (this is your AIRTABLE_TOKEN)

**Step 3: Get Base and Table IDs**
1. Visit your base → Help → API documentation  
2. Find the URL pattern: `https://api.airtable.com/v0/appXXXXXXXX/tblYYYYYYYY`
3. `appXXXXXXXX` = BASE_ID
4. `tblYYYYYYYY` = TABLE_ID

### 4. OpenRouter Setup

**Why Multiple Accounts?**
- Increases API rate limits
- Provides automatic failover if one account fails
- More total free credits available

**Step 1: Create 3 Accounts**
1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up with different email addresses:
   - youremail+1@gmail.com
   - youremail+2@gmail.com  
   - youremail+3@gmail.com

**Step 2: Generate API Keys**
1. Login to each account
2. Go to Keys section
3. Create new API key for each account
4. Save all 3 keys securely

### 5. Environment Configuration

Create a `.env` file in your project folder:

```env
# Airtable Settings
AIRTABLE_TOKEN=pat_xxxxxxxxxxxxxxxxx
AIRTABLE_BASE_ID=appxxxxxxxxxxxxxxxxx
AIRTABLE_TABLE_ID=tblxxxxxxxxxxxxxxxxx

# OpenRouter API Keys (for failover system)
OPENROUTER_API_KEY_1=sk-or-v1-xxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY_2=sk-or-v1-xxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY_3=sk-or-v1-xxxxxxxxxxxxxxxxx

# AI Model Configuration
OPENROUTER_MODEL=openai/gpt-oss-20b:free

# Security Settings
UI_PASSWORD=your_secure_password
FLASK_SECRET=your_random_secret_key_here
PORT=5000
```

**Important**: Replace all placeholder values with your actual credentials.

## Running the Application

### Development Mode
```bash
python memorybot.py
```

### Production Mode  
```bash
gunicorn memorybot:app
```

Access the application at: `http://localhost:5000`

## Usage Guide

### Adding Memories
Type statements like:
- "I completed my Python course on March 10, 2024"
- "My favorite programming language is Python"
- "I started working at Tech Corp in January 2023"

### Retrieving Information
Ask questions like:
- "When did I complete Python course?"
- "What's my favorite programming language?"
- "When did I start working?"

## API Failover System

The application uses multiple OpenRouter API keys in sequence:

1. **Primary API Key** - Used for first attempt
2. **Secondary API Key** - Used if primary fails  
3. **Tertiary API Key** - Used if secondary fails
4. **Error Response** - Only if all three keys fail

This ensures high availability and reliability.

## File Structure

```
memory-bot/
├── memorybot.py              # Main application file
├── .env                # Environment variables (keep private)
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## Deployment Options

### Local Deployment
- Run directly with Python for personal use
- Ideal for development and testing

### Cloud Deployment
**Platforms**: Render, Railway, Heroku, DigitalOcean

**Steps**:
1. Upload code to platform (exclude .env file)
2. Set environment variables in platform dashboard
3. Use `gunicorn memorybot:app` as start command
4. Deploy and access via provided URL

## Troubleshooting

### Common Issues

**"No OpenRouter keys found"**
- Verify .env file contains all three OPENROUTER_API_KEY variables
- Check .env file is in same directory as app.py

**"Airtable connection failed"**
- Confirm AIRTABLE_TOKEN has proper permissions
- Verify BASE_ID and TABLE_ID are correct
- Ensure table columns match exact names: Knowledge, Reference, Date

**"All OpenRouter keys failed"**
- Check remaining credits in OpenRouter accounts
- Verify all API keys are valid and active
- Try different model in OPENROUTER_MODEL setting

### Rate Limits
- Free OpenRouter: ~100 requests/minute per account
- With 3 accounts: ~300 requests/minute total
- Sufficient for personal use cases

## Security Best Practices

• Never commit .env file to version control
• Use strong passwords for UI_PASSWORD
• Keep API keys confidential
• Regularly rotate API keys
• Use HTTPS in production environments

## System Requirements

**Minimum**:
- Python 3.8+
- 512MB RAM
- 100MB storage space
- Internet connection

**Recommended**:
- Python 3.9+
- 1GB RAM  
- 1GB storage space
- Stable internet connection

## Technical Architecture

### Components
- **Flask Web Server** - Handles HTTP requests and responses
- **OpenRouter API** - Provides AI language processing
- **Airtable Database** - Stores structured memory data
- **Environment Management** - Secure configuration handling

### Data Flow
1. User input → Intent classification (INSERT/QUERY)
2. INSERT: Extract fields → Store in Airtable
3. QUERY: Extract keywords → Search Airtable → Generate response
4. Return result to user interface

## Support & Maintenance

### Regular Tasks
- Monitor API usage and credits
- Backup Airtable data periodically
- Update dependencies monthly
- Review and rotate API keys quarterly

### Getting Help
- Check this documentation first
- Review error messages carefully
- Verify all environment variables
- Test API keys individually if needed

## License

MIT License - Free to use and modify for personal and commercial projects.

---

**Version**: 1.0  
**Last Updated**: September 2025