(() => {
  let microstateDialog;
  let lastMicrostateTrigger;
  let lastFeedbackTrigger;

  function updateRange(range) {
    const value = Number(range.value);
    const min = Number(range.min || 0);
    const max = Number(range.max || 100);
    const position = max === min ? 0 : ((value - min) / (max - min)) * 100;
    const output = document.getElementById(range.getAttribute("aria-describedby") || "");

    range.style.setProperty("--ph-position", position.toFixed(3));
    if (output) {
      output.textContent = value.toFixed(1);
    }
  }

  function ensureMicrostateDialog() {
    if (microstateDialog) {
      return microstateDialog;
    }

    microstateDialog = document.createElement("dialog");
    microstateDialog.className = "microstate-modal";
    microstateDialog.innerHTML = `
      <div class="microstate-modal__header">
        <div class="microstate-modal__title"></div>
        <button type="button" class="microstate-modal__close" aria-label="Close enlarged microstate">&times;</button>
      </div>
      <div class="microstate-modal__body">
        <div class="microstate-modal__image"></div>
      </div>
    `;

    microstateDialog.addEventListener("click", (event) => {
      if (event.target === microstateDialog) {
        microstateDialog.close();
      }
    });

    microstateDialog.querySelector(".microstate-modal__close").addEventListener("click", () => {
      microstateDialog.close();
    });

    microstateDialog.addEventListener("close", () => {
      if (lastMicrostateTrigger && document.contains(lastMicrostateTrigger)) {
        lastMicrostateTrigger.focus();
      }
      lastMicrostateTrigger = null;
    });

    document.body.append(microstateDialog);
    return microstateDialog;
  }

  function enlargeMicrostate(button) {
    const image = button.querySelector(".microstate-image");
    if (!image) {
      return;
    }

    const dialog = ensureMicrostateDialog();
    dialog.querySelector(".microstate-modal__title").textContent =
      button.dataset.microstateTitle || "Microstate";
    dialog.querySelector(".microstate-modal__image").innerHTML = image.innerHTML;
    lastMicrostateTrigger = button;

    if (!dialog.open) {
      dialog.showModal();
    }
  }

  document.addEventListener("input", (event) => {
    if (event.target instanceof HTMLInputElement && event.target.matches("[data-ph-range]")) {
      updateRange(event.target);
    }
  });

  document.addEventListener("submit", (event) => {
    if (!(event.target instanceof HTMLFormElement) || event.target.id !== "pkasso-form") {
      return;
    }

    const pageStatus = document.getElementById("feedback-page-status");
    if (pageStatus) {
      pageStatus.innerHTML = "";
    }
  });

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }

    const feedbackButton = event.target.closest("[data-feedback-open]");
    if (feedbackButton instanceof HTMLButtonElement) {
      const dialog = document.getElementById("feedback-dialog");
      const mainSmiles = document.querySelector("#pkasso-form [name='smiles']");
      const feedbackSmiles = dialog?.querySelector("[name='smiles']");
      const feedbackStatus = document.getElementById("feedback-status");

      if (dialog instanceof HTMLDialogElement && feedbackSmiles instanceof HTMLTextAreaElement) {
        feedbackSmiles.value = mainSmiles instanceof HTMLTextAreaElement ? mainSmiles.value : "";
        if (feedbackStatus) {
          feedbackStatus.innerHTML = "";
        }
        lastFeedbackTrigger = feedbackButton;
        dialog.showModal();
      }
      return;
    }

    const feedbackClose = event.target.closest("[data-feedback-close]");
    if (feedbackClose) {
      const dialog = document.getElementById("feedback-dialog");
      if (dialog instanceof HTMLDialogElement) {
        dialog.close();
      }
      return;
    }

    const button = event.target.closest("[data-microstate-enlarge]");
    if (button instanceof HTMLButtonElement) {
      enlargeMicrostate(button);
    }
  });

  const feedbackDialog = document.getElementById("feedback-dialog");
  if (feedbackDialog instanceof HTMLDialogElement) {
    feedbackDialog.addEventListener("click", (event) => {
      if (event.target === feedbackDialog) {
        feedbackDialog.close();
      }
    });
    feedbackDialog.addEventListener("close", () => {
      if (lastFeedbackTrigger && document.contains(lastFeedbackTrigger)) {
        lastFeedbackTrigger.focus();
      }
      lastFeedbackTrigger = null;
    });
  }

  document.addEventListener("htmx:afterSwap", (event) => {
    const target = event.detail.target;
    if (!(target instanceof HTMLElement) || target.id !== "feedback-status") {
      return;
    }
    if (!target.querySelector("[data-feedback-saved]")) {
      return;
    }

    const pageStatus = document.getElementById("feedback-page-status");
    if (pageStatus) {
      pageStatus.innerHTML = target.innerHTML;
    }
    const form = target.closest("form");
    if (form instanceof HTMLFormElement) {
      form.reset();
    }
    target.innerHTML = "";
    if (feedbackDialog instanceof HTMLDialogElement) {
      feedbackDialog.close();
    }
  });

  document.querySelectorAll("[data-ph-range]").forEach(updateRange);
})();
