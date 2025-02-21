//client
/*
 * client.js
 * Final client-side JavaScript for handling WebRTC connection,
 * chat/recording controls, and independent LLM reply polling per session.
 */

// Global variables for the RTCPeerConnection, polling interval, and last reply.
var pc;
var pollingInterval;
var lastReply = "";

// Start the WebRTC connection.
function startConnection() {
  var config = { sdpSemantics: 'unified-plan' };
  if (document.getElementById('use-stun').checked) {
    config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
  }
  
  // Create the RTCPeerConnection.
  pc = new RTCPeerConnection(config);
  
  // When media tracks arrive, attach them to the appropriate elements.
  pc.addEventListener('track', function(evt) {
    if (evt.track.kind === 'video') {
      document.getElementById('video').srcObject = evt.streams[0];
    } else if (evt.track.kind === 'audio') {
      document.getElementById('audio').srcObject = evt.streams[0];
    }
  });
  
  // Update UI: Hide "Start" button, show "Stop" button.
  document.getElementById('start').style.display = 'none';
  document.getElementById('stop').style.display = 'inline-block';
  
  // Begin SDP negotiation.
  negotiate();
}

// Function to negotiate SDP with the server.
function negotiate() {
  // Add transceivers to receive video and audio.
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });
  
  pc.createOffer().then(function(offer) {
    return pc.setLocalDescription(offer);
  }).then(function() {
    // Wait for ICE gathering to complete.
    return new Promise(function(resolve) {
      if (pc.iceGatheringState === 'complete') {
        resolve();
      } else {
        function checkState() {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', checkState);
            resolve();
          }
        }
        pc.addEventListener('icegatheringstatechange', checkState);
      }
    });
  }).then(function() {
    var offer = pc.localDescription;
    return fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
    });
  }).then(function(response) {
    return response.json();
  }).then(function(answer) {
    // Store the session ID in a hidden field for later use.
    document.getElementById('sessionid').value = answer.sessionid;
    // Start polling for LLM replies now that we have a session.
    startPolling();
    // Set the remote description with the answer from the server.
    return pc.setRemoteDescription(new RTCSessionDescription({
      type: answer.type,
      sdp: answer.sdp
    }));
  }).catch(function(e) {
    alert("Negotiation error: " + e);
  });
}

// Function to stop the WebRTC connection.
function stopConnection() {
  document.getElementById('stop').style.display = 'none';
  if (pc) {
    pc.close();
    pc = null;
  }
  if (pollingInterval) {
    clearInterval(pollingInterval);
  }
}

// Clean up the connection when the page is unloaded.
window.onunload = function() { if (pc) { pc.close(); } };
window.onbeforeunload = function(e) {
  if (pc) { pc.close(); }
  e = e || window.event;
  if (e) { e.returnValue = 'Closing connection'; }
  return 'Closing connection';
};

$(document).ready(function() {
  // Handle chat form submission.
  $('#echo-form').on('submit', function(e) {
    e.preventDefault();
    var message = $('#message').val();
    var sessionId = parseInt($('#sessionid').val());
    if (!sessionId) {
      alert("Please wait while your session is being generated. The app will update with the most up-to-date information shortly.");
      return;
    }
    console.log("Sending message:", message, "sessionid:", sessionId);
    // Use type "chat" to trigger LLM processing on the server.
    fetch('/human', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: message,
        type: 'chat',
        interrupt: true,
        sessionid: sessionId
      })
    });
    $('#message').val('');
  });
  
  // Recording control: Start recording.
  $('#btn_start_record').click(function() {
    var sessionId = parseInt($('#sessionid').val());
    fetch('/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'start_record', sessionid: sessionId })
    }).then(function(response) {
      if (response.ok) {
        $('#btn_start_record').prop('disabled', true);
        $('#btn_stop_record').prop('disabled', false);
      }
    });
  });
  
  // Recording control: Stop recording.
  $('#btn_stop_record').click(function() {
    var sessionId = parseInt($('#sessionid').val());
    fetch('/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'end_record', sessionid: sessionId })
    }).then(function(response) {
      if (response.ok) {
        $('#btn_start_record').prop('disabled', false);
        $('#btn_stop_record').prop('disabled', true);
      }
    });
  });
});

// Function to poll the /llm_reply endpoint every 3 seconds.
function startPolling() {
  pollingInterval = setInterval(function() {
    var sessionId = document.getElementById('sessionid').value;
    if (!sessionId || sessionId === "0") {
      return;  // No valid session yet.
    }
    $.ajax({
      url: "/llm_reply?sessionid=" + sessionId,
      type: "GET",
      success: function(resp) {
        // Only update the UI if the reply has changed.
        if (resp.reply && resp.reply !== lastReply) {
          $("#responseArea").append("<p><strong>LLM reply:</strong> " + resp.reply + "</p>");
          lastReply = resp.reply;
        }
      },
      error: function(err) {
        console.error("Error fetching LLM reply:", err);
      }
    });
  }, 3000);
}
