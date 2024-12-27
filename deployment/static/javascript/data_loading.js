const get_circle_indicator = (value, type) => {
    if(type === "no2") {
        if(value <= 25) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-primary border border-light rounded-circle"></span>`
        } else if(value <= 50) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-success border border-light rounded-circle"></span>`
        } else if(value <= 120) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-warning border border-light rounded-circle"></span>`
        } else {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-danger border border-light rounded-circle"></span>`
        }
    } else {
        if(value <= 100) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-primary border border-light rounded-circle"></span>`
        } else if(value <= 120) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-success border border-light rounded-circle"></span>`
        } else if(value <= 160) {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-warning border border-light rounded-circle"></span>`
        } else {
            return `<span style="right:0" class="mx-3 my-1 position-absolute p-2 bg-danger border border-light rounded-circle"></span>`
        }
    }
}

const formatReadings = (no2, o3, with_indicators) => {
    // If predicted value is negative replace with zero
    const no2_value = no2 < 0 ? 0 : no2
    const o3_value = o3 < 0 ? 0 : o3

    const line_end_no2 = with_indicators ? get_circle_indicator(no2_value, "no2") : ""
    const line_end_o3 = with_indicators ? get_circle_indicator(o3_value, "o3") : ""

    // Format text fields
    const no2_text = `NO<sub>2</sub>: ${no2_value.toFixed(1)} &micro;g/m&sup3; ${line_end_no2}`
    const o3_text = `O<sub>3</sub>: ${o3_value.toFixed(1)} &micro;g/m&sup3; ${line_end_o3}`

    return `${no2_text}<br/>${o3_text}`
}

// Retrieve pollution data for the given pollution station and display the data in the corresponding station box
const retrieve_data = station_id => {
    // API key obtained from https://aqicn.org/data-platform/api/H4586/nl/
    const api_key = "69ca97a41b55b53ff1e30096c6f02b9e546a44c9"

    // Fetch current pollution data
    fetch(`https://api.waqi.info/feed/${station_id}/?token=${api_key}`)
        .then(response => {
            return response.json();
        })
        .then(data => {
            if(data.status !== "ok") {
                throw new Error('Error while retrieving the data')
            }

            // Reformat the timestamp to obtain a HH:MM format
            const time_text = `Latest update: ${data.data.time.s.split(" ")[1].split(":").slice(0, 2).join(":")}`

            // Display information in the corresponding station box
            if(document.getElementById(station_id) !== null) {
                document.getElementById(station_id).innerHTML = `${formatReadings(data.data.iaqi.no2.v, data.data.iaqi.o3.v, true)}<br/>${time_text}`
            }

            // Display raw data in admin are
            raw_data_element = document.getElementById(`raw_data_${station_id}`)
            if(raw_data_element !== null) {
                // Delete unused field
                delete data.data.forecast
                delete data.data.debug

                raw_data_element.innerHTML = JSON.stringify(data.data, null, '    ').replace("{", "  {").replace("}", "  }");
            }
        })
        .catch(error => {
            // Print error information and inform the user about the error
            console.log(error)
            if(document.getElementById(station_id) !== null) {
                document.getElementById(station_id).innerHTML = "Error while retrieving the data"
            }
        });
}

const retrieve_predictions = () => {
    fetch(`/predict`)
        .then(response => {
            return response.json();
        })
        .then(data => {
            if(data.errors.length === 0) {
                if(document.getElementById("predictions_day1") !== null) {
                    document.getElementById("predictions_day1").innerHTML = formatReadings(data.day1_no2, data.day1_o3, true)
                    document.getElementById("predictions_day2").innerHTML = formatReadings(data.day2_no2, data.day2_o3, true)
                    document.getElementById("predictions_day3").innerHTML = formatReadings(data.day3_no2, data.day3_o3, true)
                    document.getElementById("predictions_day4").innerHTML = formatReadings(data.day4_no2, data.day4_o3, true)
                }
                if (document.getElementById("predictions_date") !== null){
                    document.getElementById("predictions_date").innerHTML = data.created_at
                }
                if(document.getElementById("raw_prediction_data") !== null) {
                    document.getElementById("raw_prediction_data").innerHTML = JSON.stringify(data, null, '\t\t\t\t').replace("{", "\t\t\t{").replace("}", "\t\t\t}");
                }
            }
            
            // Handle warnings
            const warningsElement = document.getElementById("warnings");
            const warningsAlert = document.getElementById("warningAlert");
            if (warningsElement !== null) {
                if (Array.isArray(data.warnings) && data.warnings.length > 0) {
                    warningsElement.innerHTML = data.warnings.join("<br/>"); 
                    warningsAlert.style.display = "block"; 
                } else {
                    warningsAlert.style.display = "none"; 
                }
            }

            // Handle errors
            const predictionsSet = document.getElementById("predictionsSet");
            const errorsElement = document.getElementById("errors");
            const errorsAlert = document.getElementById("errorAlert");
            if (errorsElement !== null) {
                if (Array.isArray(data.errors) && data.errors.length > 0) {
                    errorsElement.innerHTML = data.errors.join("<br/>");
                    // Hide predictions if they are shown
                    if (predictionsSet != null) {
                        predictionsSet.style.display = "none";
                    }
                    errorsAlert.style.display = "block";
                } else {
                    errorsAlert.style.display = "none";
                }
            }
            
            // Success
            const succesAlert = document.getElementById("successAlert");
            if (succesAlert !== null){
                if (data.warnings.length === 0 && data.errors.length === 0) {
                    succesAlert.style.display = "block"; 
                } else {
                    succesAlert.style.display = "none"; 
                }
            }
            
        })
        .catch(error => {
            console.log(error)
        });
}

const compare_past_day_data = () => {
    Promise.all([
        fetch(`/previous_prediction`).then(response => response.json()),
        fetch(`/previous_values`).then(response => response.json()),
        fetch(`/predict`).then(response => response.json())
    ])
        .then(([prediction_data, past_day_data, prediction_data_most_recent]) => {
            if(prediction_data.errors.length === 0 && past_day_data.errors.length === 0 && prediction_data_most_recent.errors.length === 0) {
                const predicted_no2 = Math.max(prediction_data.day1_no2, 0)
                const predicted_o3 = Math.max(prediction_data.day1_o3, 0)
                const predicted_no2_most_recent = Math.max(prediction_data_most_recent.day1_no2, 0)
                const predicted_o3_most_recent = Math.max(prediction_data_most_recent.day1_o3, 0)

                const actual_no2 = past_day_data.no2
                const actual_o3 = past_day_data.o3
                const actual_no2_most_recent = past_day_data.no2_most_recent
                const actual_o3_most_recent = past_day_data.o3_most_recent

                const delta_no2 = Math.abs(actual_no2 - predicted_no2)
                const delta_o3 = Math.abs(actual_o3 - predicted_o3)
                const delta_no2_most_recent = Math.abs(actual_no2_most_recent - predicted_no2_most_recent)
                const delta_o3_most_recent = Math.abs(actual_o3_most_recent - predicted_o3_most_recent)

                document.getElementById("yesterday_no2_error").innerHTML =
                  `Actual NO<sub>2</sub>: ${actual_no2.toFixed(1)} &micro;g/m&sup3;<br/>Predicted NO<sub>2</sub>: ${predicted_no2.toFixed(1)} &micro;g/m&sup3;<br/>Difference: ${delta_no2.toFixed(1)} &micro;g/m&sup3;`
                document.getElementById("yesterday_o3_error").innerHTML =
                  `Actual O<sub>3</sub>: ${actual_o3.toFixed(1)} &micro;g/m&sup3;<br/>Predicted O<sub>3</sub>: ${predicted_o3.toFixed(1)} &micro;g/m&sup3;<br/>Difference: ${delta_o3.toFixed(1)} &micro;g/m&sup3;`
                document.getElementById("today_no2_error").innerHTML =
                  `Actual NO<sub>2</sub>: ${actual_no2_most_recent.toFixed(1)} &micro;g/m&sup3;<br/>Predicted NO<sub>2</sub>: ${predicted_no2_most_recent.toFixed(1)} &micro;g/m&sup3;<br/>Difference: ${delta_no2_most_recent.toFixed(1)} &micro;g/m&sup3;`
                document.getElementById("today_o3_error").innerHTML =
                  `Actual O<sub>3</sub>: ${actual_o3_most_recent.toFixed(1)} &micro;g/m&sup3;<br/>Predicted O<sub>3</sub>: ${predicted_o3_most_recent.toFixed(1)} &micro;g/m&sup3;<br/>Difference: ${delta_o3_most_recent.toFixed(1)} &micro;g/m&sup3;`
            }
        })
        .catch(error => {
            console.log(error)
        });
}

const set_date_text = (element_id, days_after_today) => {
    const today = new Date();
    const new_date = new Date();
    new_date.setDate(today.getDate() + days_after_today);
    
    document.getElementById(element_id).innerHTML = `${new_date.toLocaleDateString()}`
}

const pollution_stations = ["@6332", "@4584", "@4585", "@4586"]

// For each station, if the corresponding HTML box exists, load the data
pollution_stations.forEach(station_id => {
    retrieve_data(station_id)
});

retrieve_predictions()

if(document.getElementById("day3_name") !== null) {
    set_date_text("day3_name", 2)
}

if(document.getElementById("day4_name") !== null) {
    set_date_text("day4_name", 3)
}

if(document.getElementById("yesterday_no2_error") !== null) {
    compare_past_day_data()
}
