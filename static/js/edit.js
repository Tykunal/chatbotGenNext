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
      document.getElementById("userId").value = ticketData.userid || "";
      document.getElementById("application").value =
        ticketData.application || "";
      document.getElementById("problemType").value =
        ticketData.problem_type || "";
      document.getElementById("problemDescription").value =
        ticketData.description || "";

      const statusRadios = document.getElementsByName("status");
      let i = 0;
      while (i < 3) {
        if (statusRadios[i].value === ticketData.status) {
          statusRadios[i].checked = true;
          break;
        } else {
          i++;
        }
      }
      // Storing for comparision in data change
      originalData.problemType = ticketData.problem_type || "";
      originalData.problemDescription = ticketData.description || "";
      originalData.status = ticketData.status || "";
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
