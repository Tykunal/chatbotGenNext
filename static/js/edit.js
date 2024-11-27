const originalData = {
  problemType: "",
  problemDescription: "",
  status: "",
};
const urlParams = new URLSearchParams(window.location.search);
const ticketNumber = urlParams.get("ticketNumber");
document.getElementById("ticketNumber").value = ticketNumber;

async function getPreFilledFormData() {
  try {
    const response = await fetch("http://127.0.0.1:5000/getEditFormData", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ticketNumber }),
    });

    if (response.ok) {
      const ticketData = await response.json();
      // console.log(ticketData.Status);
      document.getElementById("userId").value = ticketData.User_ID || "";
      document.getElementById("application").value =
        ticketData.Application || "";
      document.getElementById("problemType").value =
        ticketData.Problem_Type || "";
      document.getElementById("problemDescription").value =
        ticketData.Problem_Description || "";

      const statusRadios = document.getElementsByName("status");
      let i = 0;
      while (i < 3) {
        if (statusRadios[i].value === ticketData.Status) {
          statusRadios[i].checked = true;
          break;
        } else {
          i++;
        }
      }
      // Storing for comparision in data change
      originalData.problemType = ticketData.Problem_Type || "";
      originalData.problemDescription = ticketData.Problem_Description || "";
      originalData.status = ticketData.Status || "";
    } else {
      const errorData = await response.json();
      alert(errorData.error || "Failed to fetch ticket details");
    }
  } catch (error) {
    console.error("Error fetching ticket details:", error);
    alert("An error occurred while fetching ticket details.");
  }
}
// handling form submission
document
  .querySelector("form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const problemType = document.getElementById("problemType").value;
    const problemDescription =
      document.getElementById("problemDescription").value;

    const statusRadios = document.getElementsByName("status");
    let status = "";
    for (const radio of statusRadios) {
      if (radio.checked) {
        status = radio.value;
        break;
      }
    }

    const data = {
      ticketNumber,
    };

    // Compare current field values to pre-filled data
    if (problemType !== originalData.problemType)
      data.problemType = problemType;
    if (problemDescription !== originalData.problemDescription)
      data.problemDescription = problemDescription;
    if (status !== originalData.status) data.status = status;

    // If no changes were made, inform the user and don't send the request
    if (Object.keys(data).length === 1) {
      alert("No changes detected.");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/editTicket", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        alert(result.message || "Ticket updated successfully!");
        window.location.href = "/admin";
      } else {
        const errorData = await response.json();
        alert(errorData.error || "Failed to update the ticket.");
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      alert("An error occurred while submitting the form.");
    }
  });

getPreFilledFormData();
