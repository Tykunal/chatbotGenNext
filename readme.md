# Chatbot

A chatbot designed for the **GenNext project** that handles user queries related to applications, assigns tickets for unresolved issues, and updates user data in the backend.

---

## Features
- **Query Handling**: Efficiently processes user queries and provides appropriate responses.
- **Authentication**: Allowing users to SignUp/ Login and update the details in backend.
- **Ticket Management**: Automatically generates and assigns tickets for issues that cannot be resolved by the chatbot.
- **Backend Integration**: Updates user information in the backend system.
- **Admin Portal**: Provides a admin page with all tickets generated with the ability to edit and delete them.

---

## Prerequisites
Before running the chatbot, ensure you have the following installed:
1. **Python** (Version >= 3.8 recommended)
2. Required Python libraries:
   - `flask`
   - `nltk`
   - `torch`
   - `numpy`

Install all dependencies using:
```bash
pip install -r requirements.txt
