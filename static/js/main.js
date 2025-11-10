let predicted = "Choose an image to predict"
let video = document.getElementById('video')
var form = document.getElementById("img-form");
var realData;
let canvas = document.createElement('canvas');
let context = canvas.getContext('2d');
let predictionHistory = [];
const maxHistoryItems = 10;
let isCameraOn = false;
let recognitionInterval = null;
let consecutivePredictions = []; // To track consistency
const maxConsecutive = 3; // Number of consistent predictions needed
let processingImage = false; // Flag to prevent overlapping predictions

// Function to set recognition mode
function setMode(mode) {
    // Update UI
    if (mode === 'letter') {
        $('#letter-mode-btn').addClass('active').removeClass('btn-outline-primary').addClass('btn-primary');
        $('#phrase-mode-btn').removeClass('active').removeClass('btn-primary').addClass('btn-outline-primary');
    } else {
        $('#phrase-mode-btn').addClass('active').removeClass('btn-outline-primary').addClass('btn-primary');
        $('#letter-mode-btn').removeClass('active').removeClass('btn-primary').addClass('btn-outline-primary');
    }
    
    // Send mode to server
    updateRecognitionSettings();
}

// Function to update recognition settings on the server
function updateRecognitionSettings() {
    const mode = $('#letter-mode-btn').hasClass('active') ? 'letter' : 'phrase';
    const classifier = $('input[name="classifier-type"]:checked').val();
    
    $.ajax({
        url: '/set_mode',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ mode: mode, classifier: classifier }),
        success: function(response) {
            console.log('Recognition settings updated:', response);
        },
        error: function(error) {
            console.error('Error updating recognition settings:', error);
        }
    });
}

$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result-section').hide();
    $('.take-section').hide();
    $('#btn-predict-img').hide();
    $('#toggle-camera').html('<i class="fas fa-camera me-2"></i>Start Camera');
    $('#video').hide();
    $('#video-placeholder').show();
    
    // Set up classifier type change listener
    $('input[name="classifier-type"]').change(function() {
        updateRecognitionSettings();
    });

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    window.stopVideo = function() {
        if (video.srcObject) {
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
            isCameraOn = false;
            $('#toggle-camera').html('<i class="fas fa-camera me-2"></i>Start Camera');
            $('#prediction').text('-');
            $('#confidence-bar').css('width', '0%');
            $('#video').hide();
            $('#video-placeholder').show();
        }
    }

    window.startVideo = function() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    video.srcObject = stream;
                    video.play();
                    isCameraOn = true;
                    $('#toggle-camera').html('<i class="fas fa-camera-slash me-2"></i>Stop Camera');
                    $('#video-placeholder').hide();
                    $('#video').show();
                    
                    // Set up canvas dimensions
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    
                    // Start capturing frames for prediction
                    startRecognition();
                })
                .catch(function(error) {
                    console.error("Error accessing camera:", error);
                    alert("Error accessing camera. Please make sure you have granted camera permissions.");
                });
        } else {
            alert("Your browser doesn't support camera access.");
        }
    }

    function takeSnapshot() {
      var img = document.querySelector('.img-display').querySelector('img')
      console.log(img)
      var width = video.offsetWidth
        , height = video.offsetHeight;

      canvas.width = width;
      canvas.height = height;

      context.drawImage(video, 0, 0, width, height);

      img.src = canvas.toDataURL('image/png');
      document.querySelector('.img-display').appendChild(img);
      $('#btn-predict-img').show();

      var ImageURL = img.src; // 'photo' is your base64 image
      // Split the base64 string in data and contentType
      var block = ImageURL.split(";");
      // Get the content type of the image
      var contentType = block[0].split(":")[1];// In this case "image/gif"
      // get the real base64 content of the file
      realData = block[1].split(",")[1];
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result-section').hide();
        $('#btn-predict-img').hide();
        readURL(this);
    });

    $(".upload-btn").click(function () {
        $('.take-section').hide();

    });

    $(".click-photo").click(function () {
        $('.image-section').hide();
        $('#btn-predict').hide();
        $('#result').text('');
        $('#result-section').hide();
        $('.take-section').show();
    });

    $("#btn-clear").click(function () {
        $('.image-section').hide();
        $('#btn-predict').hide();
        $('#result').text('');
        $('#result-section').hide();
    });

    $("#click-photo").click(function () {
        $('.video-section').show();
    });

    $(".btn-take-pic").click(function () {
        takeSnapshot()
    });

    $(".start-btn").click(function () {
        startVideo()
    });

    $(".stop-btn").click(function () {
        stopVideo()
        document.querySelector(".btn-take-pic").disabled = true;
        document.querySelector(".stop-btn").disabled = true;
        document.querySelector(".start-btn").disabled = false;
    });

    $("#speaker").click(function () {
        var msg = new SpeechSynthesisUtterance();
        msg.text = predicted;
        window.speechSynthesis.speak(msg);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        console.log(form_data)

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result-section').fadeIn(600);
                $('#result').text(data);
                predicted = data
                console.log('Success!');
            },
        });
    });

    // Predict
    $('#btn-predict-img').click(function () {
        // Show loading animation
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict-img',
            data: realData,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result-section').fadeIn(600);
                $('#result').text(data);
                predicted = data
                console.log('Success!');
            },
        });

    });

    window.toggleCamera = function() {
        if (isCameraOn) {
            stopVideo();
        } else {
            startVideo();
        }
    }

    window.clearHistory = function() {
        predictionHistory = [];
        updateHistoryDisplay();
    }

});

// Start the recognition loop with enhanced temporal smoothing
function startRecognition() {
    // Clear any existing interval
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
    }
    
    // Reset tracking variables
    consecutivePredictions = [];
    processingImage = false;
    predictionBuffer = [];
    lastStablePrediction = null;
    
    recognitionInterval = setInterval(() => {
        if (!isCameraOn || !video.srcObject || processingImage) {
            return; // Skip if camera is off or already processing
        }
        
        processingImage = true;
        
        // Draw the current video frame to canvas
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get the image data with higher quality for better detection
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        
        // Send to server for prediction with temporal smoothing
        predictSignWithDebounce(imageData);
    }, 500);
}

// Use debounce to prevent too many requests
function predictSignWithDebounce(imageData) {
    predictSign(imageData);
}

// Send image to server for prediction
async function predictSign(imageData) {
    try {
        // Show processing indicator
        const predictionElement = document.getElementById('prediction');
        if (predictionElement) {
            predictionElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }
        
        const response = await fetch('/predict-img', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update with stabilization (only update UI after consistent predictions)
            handleStablePrediction(data.prediction, data.confidence);
        } else {
            console.error("Prediction error:", data.error);
            // Display error to user
            if (predictionElement) {
                predictionElement.textContent = data.error === "Hand not detected" ? 
                    "No hand detected" : data.error;
                
                // Reset confidence for errors
                const confidenceBar = document.getElementById('confidence-bar');
                if (confidenceBar) {
                    confidenceBar.style.width = "0%";
                    confidenceBar.style.backgroundColor = "#dc3545";
                }
            }
            // Clear consecutive predictions for stability
            consecutivePredictions = [];
        }
    } catch (error) {
        console.error('Error during prediction:', error);
        const predictionElement = document.getElementById('prediction');
        if (predictionElement) {
            predictionElement.textContent = "Error processing image";
        }
    } finally {
        // Always reset processing flag when done
        processingImage = false;
    }
}

// Handle prediction with stabilization
function handleStablePrediction(prediction, confidence) {
    // Ignore predictions with low confidence
    if (confidence < 0.70) {
        return;
    }
    
    // Add to consecutive predictions with confidence weight
    consecutivePredictions.push({ prediction, confidence });
    
    // Keep only the last few predictions
    if (consecutivePredictions.length > maxConsecutive) {
        consecutivePredictions.shift();
    }
    
    // Calculate weighted votes for each prediction
    const votes = {};
    let totalWeight = 0;
    
    consecutivePredictions.forEach(pred => {
        const weight = pred.confidence * pred.confidence; // Square the confidence for more weight on high-confidence predictions
        votes[pred.prediction] = (votes[pred.prediction] || 0) + weight;
        totalWeight += weight;
    });
    
    // Find the prediction with highest weighted votes
    let maxVotes = 0;
    let stablePrediction = null;
    let stabilityScore = 0;
    
    for (const [pred, voteWeight] of Object.entries(votes)) {
        const normalizedWeight = voteWeight / totalWeight;
        if (normalizedWeight > maxVotes) {
            maxVotes = normalizedWeight;
            // Get the recognition mode from the active button
            const recognitionMode = $('#letter-mode-btn').hasClass('active') ? 'letter' : 'phrase';
            // For letter mode, the prediction is already the correct letter from the keypoint classifier
            stablePrediction = pred;
            stabilityScore = normalizedWeight;
        }
    }
    
    // Only update if we have a stable prediction with high confidence
    if (stablePrediction && stabilityScore > 0.6 && consecutivePredictions.length >= 2) {
        // Calculate average confidence of the stable prediction
        const stablePredictions = consecutivePredictions.filter(p => p.prediction === stablePrediction);
        const avgConfidence = stablePredictions.reduce((sum, p) => sum + p.confidence, 0) / stablePredictions.length;
        
        updatePrediction(stablePrediction, avgConfidence);
        
        // Only add to history if it's different from the last entry and has high stability
        const lastHistoryItem = predictionHistory.length > 0 ? predictionHistory[0].prediction : null;
        if (lastHistoryItem !== stablePrediction && stabilityScore > 0.7) {
            addToHistory(stablePrediction, avgConfidence);
        }
    }
}

// Find the most frequent item in an array
function findMostFrequent(arr) {
    const counts = {};
    let maxValue = null;
    let maxCount = 0;
    
    for (const item of arr) {
        counts[item] = (counts[item] || 0) + 1;
        if (counts[item] > maxCount) {
            maxCount = counts[item];
            maxValue = item;
        }
    }
    
    return { value: maxValue, count: maxCount };
}

// Update the prediction display
function updatePrediction(prediction, confidence) {
    const predictionElement = document.getElementById('prediction');
    const confidenceBar = document.getElementById('confidence-bar');
    
    predictionElement.textContent = prediction;
    confidenceBar.style.width = `${confidence * 100}%`;
    
    // Set color based on confidence
    if (confidence > 0.85) {
        confidenceBar.style.backgroundColor = "#28a745"; // Green for high confidence
    } else if (confidence > 0.75) {
        confidenceBar.style.backgroundColor = "#ffc107"; // Yellow for medium confidence
    } else {
        confidenceBar.style.backgroundColor = "#dc3545"; // Red for lower confidence
    }
    
    // Add animation class
    predictionElement.classList.add('fade-in');
    setTimeout(() => {
        predictionElement.classList.remove('fade-in');
    }, 300);
}

// Add prediction to history
function addToHistory(prediction, confidence) {
    // Create new history item
    const historyItem = document.createElement('div');
    historyItem.className = 'prediction-item';
    historyItem.innerHTML = `
        <span>${prediction}</span>
        <span>${(confidence * 100).toFixed(1)}%</span>
    `;
    
    // Add to beginning of history
    predictionHistory.unshift({ prediction, confidence });
    
    // Limit history size
    if (predictionHistory.length > maxHistoryItems) {
        predictionHistory.pop();
    }
    
    // Update history display
    updateHistoryDisplay();
}

// Update the history display
function updateHistoryDisplay() {
    const historyContainer = document.getElementById('prediction-history');
    if (!historyContainer) return;
    
    historyContainer.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'prediction-item';
        historyItem.innerHTML = `
            <span>${item.prediction}</span>
            <span>${(item.confidence * 100).toFixed(1)}%</span>
        `;
        historyContainer.appendChild(historyItem);
    });
}