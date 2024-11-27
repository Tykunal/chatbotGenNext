async function fetchTickets() {
  try {
    const response = await fetch("http://127.0.0.1:5000/adminData");
    const tickets = await response.json();

      console.log(tickets[0]);

    const ticketsContainer = document.getElementById("tickets-container");
    ticketsContainer.innerHTML = "";

    if (tickets && tickets.length > 0) {
      const table = document.createElement("table");

      // Table headers
      table.innerHTML = `
                            <thead>
                                <tr>
                                    <th>Ticket Number</th>
                                    <th>User ID</th>
                                    <th>Application</th>
                                    <th>Problem Type</th>
                                    <th>Problem Description</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                            </tbody>
                        `;

      const tbody = table.querySelector("tbody");

      tickets.forEach((ticket) => {
        const row = document.createElement("tr");

        // Create cells for each ticket property
        row.innerHTML = `
                                <td>${ticket["ticket_number"]}</td>
                                <td>${ticket["userid"]}</td>
                                <td>${ticket["application"]}</td>
                                <td>${ticket["problem_Type"]}</td>
                                <td>${ticket["description"]}</td>
                                <td>${ticket["status"]}</td>
                                <td>
                                  <span class="icon" onclick="editTicket('${ticket["ticket_number"]}')">
                                    <i class="fas fa-edit"></i> <!-- Edit Icon -->
                                  </span>
                                  <span class="icon" onclick="deleteTicket('${ticket["ticket_number"]}')">
                                    <i class="fas fa-trash-alt"></i> <!-- Delete Icon -->
                                  </span>
                                </td>
                            `;
        tbody.appendChild(row);
      });

      ticketsContainer.appendChild(table);
    } else {
      ticketsContainer.innerHTML =
        "<p class='no-tickets'>No tickets found.</p>";
    }
  } catch (error) {
    console.error("Error fetching tickets:", error);
    document.getElementById("tickets-container").innerHTML =
      "<p class='no-tickets'>Error: Unable to fetch tickets.</p>";
  }
}

async function editTicket(ticketNumber) {
  window.location.href = `http://127.0.0.1:5000/edit?ticketNumber=${encodeURIComponent(
    ticketNumber
  )}`;
}

async function deleteTicket(ticketNumber) {
  if (confirm(`Are you sure you want to delete Ticket ${ticketNumber}?`)) {
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/deleteTicket?ticketNumber=${ticketNumber}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        const result = await response.json();
        alert(result.message);
        fetchTickets();
      } else {
        const error = await response.json();
        alert(error.error);
      }
    } catch (error) {
      console.error("Error deleting ticket:", error);
      alert("Failed to delete ticket. Please try again later.");
    }
  }
}
fetchTickets();
