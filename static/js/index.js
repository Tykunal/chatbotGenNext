isLoggedIn(); //checking if the session exists in backend
const messagesDiv = document.getElementById("chatbot-messages");
const userInput = document.getElementById("user-input");
const sendBtn = document.querySelector(".chatbot-input button");
let isChatbotOpened = false;
let isUserRegistered = false;

function toggleChatbot() {
  const chatbotWindow = document.getElementById("chatbot-window");
  chatbotWindow.style.display =
    chatbotWindow.style.display === "none" ? "flex" : "none";
  if (!isChatbotOpened && chatbotWindow.style.display === "flex") {
    isChatbotOpened = true;
    startChat();
  }
}

// Send message when button is clicked
async function sendMessage() {
  // console.log("Send Message function is getting called.");
  const userMessage = userInput.value.trim();
  if (userMessage === "") return;
  // console.log("UserMessage: ", userMessage);
  addMessage(userMessage, "user-message");
  userInput.value = "";
  // userInput.disabled = true;
  // sendBtn.disabled = true;

  // Send user message to backend
  await fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: userMessage,
      isUserRegistered,
    }),
  })
    .then((response) => response.json())
    .then(async (data) => {
      if (data.reply) {
        addMessage(data.reply, "bot-message");

        // Check if there is a question in the response
        if (data.question) {
          addMessage(data.question, "bot-message");

          // Enable user input to get the response
          userInput.disabled = false;
          sendBtn.disabled = false;

          const userResponse = await waitForUserInput();
          console.log("UserResponse inside the chat", userResponse);
          addMessage(userResponse, "user-message");

          if (userResponse.toLowerCase() === "yes") {
            // Call the raiseTicket API
            await fetch("http://127.0.0.1:5000/raiseTicket", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                tag: data.tag,
                description: userMessage,
              }),
            })
              .then((response) => response.json())
              .then((ticketData) => {
                addMessage(ticketData.reply, "bot-message");
              })
              .catch((error) => {
                addMessage("Error: Unable to raise ticket.", "bot-message");
                console.error("Error:", error);
              });
          } else {
            addMessage(
              "Hope your problem is solved. Thanks for visiting!",
              "bot-message"
            );
          }
        } else if (isUserRegistered) {
          // Scenario we still raise the ticket based on user confirmation.

          // Ask the user if they want to raise a ticket
          userInput.disabled = false;
          sendBtn.disabled = false;

          const userResponse = await waitForUserInput();
          console.log("UserResponse after no question", userResponse);
          addMessage(userResponse, "user-message");

          if (userResponse.toLowerCase() === "yes") {
            await fetch("http://127.0.0.1:5000/raiseTicket", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                tag: data.tag,
                description: userMessage,
              }),
            })
              .then((response) => response.json())
              .then((ticketData) => {
                addMessage(ticketData.reply, "bot-message");
              })
              .catch((error) => {
                addMessage("Error: Unable to raise ticket.", "bot-message");
                console.error("Error:", error);
              });
          } else {
            addMessage(
              "Hope your problem is solved. Thanks for visiting!",
              "bot-message"
            );
          }
        }
      } else {
        addMessage("Error: No response from server.", "bot-message");
      }

      userInput.disabled = false;
      sendBtn.disabled = false;

      if (!isUserRegistered) {
        requestUserDetails();
      }
    })
    .catch((error) => {
      addMessage("Error: Unable to connect to server.", "bot-message");
      console.error("Error:", error);

      userInput.disabled = false;
      sendBtn.disabled = false;
    });
}
userInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    event.preventDefault();
    sendBtn.click();
  }
});
// Add message to the chat window
function addMessage(message, type) {
  const messageElement = document.createElement("div");
  messageElement.classList.add("message", type);
  messageElement.textContent = message;
  messagesDiv.appendChild(messageElement);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
async function checkUserRegistration(email, phone) {
  try {
    const response = await fetch("http://127.0.0.1:5000/checkUser", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, phone }),
    });
    const data = await response.json();
    if (data.success) {
      isUserRegistered = true;
      addMessage(
        `Welcome Back! ${data.name}, you're working on ${data.application}`,
        "bot-message"
      );
      addMessage("What may i help you with today?", "bot-message");
    } else {
      isUserRegistered = false;
      addMessage(`Sorry, ${data.message}`, "bot-message");
    }
  } catch (error) {
    console.error("Error checking user registration:", error);
  }
}

// Start a conversation
function startChat() {
  addMessage(
    "Hello! I'm your Helpdesk bot. How can I assist you today?",
    "bot-message"
  );
}

// Request user details (email, phone, name)
async function requestUserDetails() {
  if (!isUserRegistered) {
    let email, phone, application, userid, name;

    let isValidEmail = false;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    while (!isValidEmail) {
      addMessage("Please provide your email address.", "bot-message");
      await waitForUserInput().then((input) => {
        email = input.trim();
        if (!emailRegex.test(email)) {
          addMessage(
            "Please enter a valid email address (e.g., user@example.com).",
            "bot-message"
          );
        } else {
          isValidEmail = true;
          addMessage(`Your email: ${email}`, "user-message");
        }
      });
    }

    let isValidPhone = false;
    const phoneRegex = /^\d{10}$/;
    while (!isValidPhone) {
      addMessage("Please provide your phone number:", "bot-message");
      await waitForUserInput().then((input) => {
        phone = input.trim();
        if (!phoneRegex.test(phone)) {
          addMessage(
            "Please enter a valid 10-digit phone number.",
            "bot-message"
          );
        } else {
          isValidPhone = true;
          addMessage(`Your phone number: ${phone}`, "user-message");
        }
      });
    }

    await checkUserRegistration(email, phone);
    sendBtn.onclick = sendMessage;
    if (!isUserRegistered) {
      // Choosing correct application ID
      let isApplication = false;
      const validApplications = ["pensire", "vastuteq", "procu"];
      while (!isApplication) {
        addMessage(
          "Please select application (Pensire, Vastuteq, Procu):",
          "bot-message"
        );
        await waitForUserInput().then((input) => {
          application = input.trim().toLowerCase();

          if (!validApplications.includes(application)) {
            isApplication = false;
            addMessage(
              "Please choose a valid application ID (Pensire, Vastuteq, Procu)!",
              "bot-message"
            );
          } else {
            isApplication = true;
            addMessage(`Your Application: ${application}`, "user-message");
          }
        });
      }
      // unique id concept
      let isUniqueId = false;
      while (!isUniqueId) {
        addMessage("Please enter a unique userid:", "bot-message");
        await waitForUserInput().then((input) => {
          userid = input.trim();
          addMessage(`Your id: ${userid}`, "user-message");
        });

        await fetch("http://127.0.0.1:5000/uniqueId", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ userid }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.success) {
              addMessage(data.message, "bot-message");
              isUniqueId = true;
            } else {
              addMessage(data.message, "bot-message");
            }
          });
      }
      let isValidName = false;
      const nameRegex = /^[a-zA-Z\s-]+$/;
      while (!isValidName) {
        addMessage("Please provide your name.", "bot-message");
        await waitForUserInput().then((input) => {
          name = input.trim();
          if (!nameRegex.test(name)) {
            addMessage(
              "Please enter a valid name (letters, spaces, or hyphens only).",
              "bot-message"
            );
          } else {
            isValidName = true;
            addMessage(`Your name: ${name}`, "user-message");
          }
        });
      }

      // Save user details to backend
      await fetch("http://127.0.0.1:5000/registerUser", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, phone, name, userid, application }),
      })
        .then((response) => response.json())
        .then(async (data) => {
          if (data.success) {
            isUserRegistered = true;
            addMessage(
              "You are now registered! You can start chatting.",
              "bot-message"
            );
            addMessage("What may i help you with today?", "bot-message");
            sendBtn.onclick = sendMessage;
            await getUser();
            document.getElementById("logOutBtn").style.display = "block";
          } else {
            addMessage(
              "Sorry, there was an error registering your details.",
              "bot-message"
            );
          }
        });
    }
    await getUser();
    document.getElementById("logOutBtn").style.display = "block";
  }
}

function waitForUserInput() {
  return new Promise((resolve) => {
    sendBtn.onclick = () => {
      const input = userInput.value.trim();
      console.log("User input received:", input); // Debugging
      userInput.value = "";
      resolve(input);
    };
  });
}
async function isLoggedIn() {
  await fetch("http://127.0.0.1:5000/isLogged")
    .then((response) => response.json())
    .then((data) => {
      if (data.isLoggedIn) {
        isUserRegistered = true;
        document.getElementById("logOutBtn").style.display = "block";
      } else {
        isUserRegistered = false;
      }
    })
    .catch((error) => console.error("Error fetching currentUser:", error));
}
async function getUser() {
  await fetch("http://127.0.0.1:5000/get_currentUser")
    .then((response) => response.json())
    .then((data) => {
      const currentUser = data.currentUser;
      if (currentUser) {
        console.log(currentUser);
        document.getElementById("adminButton").style.display = "block";
      }
    })
    .catch((error) => console.error("Error fetching currentUser:", error));
}
async function handleLogOut() {
  try {
    const response = await fetch("http://127.0.0.1:5000/logOut");

    if (response.ok) {
      const data = await response.json();
      alert(data.message);
      // Optionally redirect to login page or refresh the page
      window.location.href = '/';
    } else {
      const errorData = await response.json();
      alert(errorData.error || "Error logging out");
    }
  } catch (error) {
    console.error("Error Logging Out:", error);
    alert("An unexpected error occurred while logging out.");
  }
}

window.onload = async function () {
  await isLoggedIn();
  await getUser();
}; //This will keep the button visible if session exists.
