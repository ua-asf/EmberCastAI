#![allow(non_snake_case)]

use dioxus::{
    document::{Style, Title},
    prelude::*,
};
use tokio::process::Command;

fn main() {
    dioxus::launch(App);
}

pub static USERNAME: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static PASSWORD: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static WKT_STRING: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static OUTPUT_FILES: GlobalSignal<Vec<String>> = GlobalSignal::new(Vec::new);
pub static STATUS_MESSAGE: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static INDEX: GlobalSignal<usize> = GlobalSignal::new(|| 0);

pub static THROBBER: Asset = asset!("assets/throbber.svg");

#[component]
pub fn App() -> Element {
    rsx! {
        Title { "EmberCast AI" }
        // Stylesheet
        // Black background with white text
        Style {
            r#"@font-face {{
                font-family: 'Pixel';
                src: url('assets/fonts/pixeloid/PixeloidSans-mLxMm.ttf') format('truetype');
                font-weight: 400;
                font-display: swap;
                font-style: normal;
            }}

            body {{
                background-color: #020202;
                color: #FEFEFE;
                margin: 0;
                font-family: 'Pixel';
                font-size: 16px;
             }}

            button {{
                font-family: 'Pixel';
                font-size: 18px;
                border-radius: 0px;
            }}

            input {{
                font-family: 'Pixel';
            }}"#
        }
        div { style: "text-align: center;
            height: 100%;
            width: 100vw;
            display: grid;
            gap: 20px;
            grid-template-columns: auto 5px 1fr;
            height: 100%;
            flex: 1,
            margin: 0px 0px;
            height: 100vh;
            display: flex;
            align-items: center;",
            UIinputs {}
            Separator {}
            RenderImage {}
        }
    }
}

#[component]
fn UIinputs() -> Element {
    let mut username_error = use_signal(|| false);
    let mut password_error = use_signal(|| false);
    let mut wkt_string_error = use_signal(|| false);
    let mut button_clickable = use_signal(|| true);

    rsx! {
        div { style: "display: flex; flex-direction: column; flex: 0 1 auto; justify-content: center; padding: 20px; min-height: 50%; max-height: 80%",
            div {
                p { "Earthdata Username" }
                input {
                    style: format!("border-color: {}", if username_error() { "red" } else { "white" }),
                    oninput: move |e| {
                        *USERNAME.write() = Some(e.value().clone());
                        username_error.set(false);
                    },
                    value: USERNAME(),
                }
            }

            div {
                p { "Earthdata Password" }
                input {
                    style: format!(
                        "letter-spacing: 2pt; border-color: {}",
                        if password_error() { "red" } else { "white" },
                    ),
                    r#type: "password",
                    oninput: move |e| {
                        *PASSWORD.write() = Some(e.value().clone());
                        password_error.set(false);
                    },
                    value: PASSWORD(),
                }
            }

            div {
                p { "WKT String" }
                input {
                    style: format!("border-color: {}", if wkt_string_error() { "red" } else { "white" }),
                    oninput: move |e| {
                        *WKT_STRING.write() = Some(e.value().clone());
                        wkt_string_error.set(false);
                    },
                    value: WKT_STRING(),
                }
            }

            div { style: "margin-top: 20px; display: flex; justify-content: center;",
                button {
                    style: "width: 100%; padding: 5px;",
                    onclick: move |_| {
                        let mut errors = false;
                        if USERNAME.read().clone().is_none_or(|v| v.is_empty()) {
                            username_error.set(true);
                            errors = true;
                        }
                        if PASSWORD.read().clone().is_none_or(|v| v.is_empty()) {
                            password_error.set(true);
                            errors = true;
                        }
                        if WKT_STRING.read().clone().is_none_or(|v| v.is_empty()) {
                            wkt_string_error.set(true);
                            errors = true;
                        }
                        if errors {
                            return;
                        }
                        button_clickable.set(false);
                        let date_format_str = "%Y-%m-%dT%H:%M:%S.%3f";
                        let formatted_date: String = chrono::Local::now()
                            .format(date_format_str)
                            .to_string();
                        println!("Formatted date: {}", formatted_date);

                        if !OUTPUT_FILES.read().is_empty() {
                            *INDEX.write() = OUTPUT_FILES.read().len() - 1;
                        }

                        OUTPUT_FILES.write().push(THROBBER.to_string());
                        spawn(async move {
                            run_model(
                                    &USERNAME().unwrap_or_default(),
                                    &PASSWORD().unwrap_or_default(),
                                    &WKT_STRING().unwrap_or_default(),
                                    &formatted_date,
                                )
                                .await;
                            button_clickable.set(true);
                        });
                    },
                    disabled: !button_clickable(),
                    if button_clickable() {
                        "Run Model"
                    } else {
                        "Loading..."
                    }
                }
            }
        }
    }
}

#[component]
fn Separator() -> Element {
    rsx! {
        div { style: "width: 5px; height: 100%; background-color: #FFF;" }
    }
}

#[component]
fn RenderImage() -> Element {
    let files_count = OUTPUT_FILES.read().len();

    let index = *INDEX.read();

    rsx! {
        div { style: "padding-left: 20px; padding-right: 20px; display: flex; flex-direction: column; justify-content: start; align-items: center; height: 100vh; width: 100%; overflow: hidden",
            // Image navigation buttons
            div { style: "display: flex; justify-content: center; align-items: center; align-self: start; width: 100%; gap: 10px; padding-bottom: 10px; padding-top: 20px;",
                // Previous/Decrement button
                button {
                    disabled: index == 0,
                    onclick: move |_| {
                        if index > 0 {
                            *INDEX.write() = INDEX() - 1;
                        }
                    },
                    "← Previous"
                }

                // Count
                p { style: "margin: 0; padding: 2px;",
                    if files_count > 0 {
                        "{index + 1}/{files_count}"
                    } else {
                        "0/0"
                    }
                }

                // Next/Increment button
                button {
                    disabled: index + 1 >= files_count,
                    onclick: move |_| {
                        if index < files_count - 1 {
                            *INDEX.write() = index + 1;
                        }
                    },
                    "Next →"
                }
            }

            // Render the selected image if any are available
            if !OUTPUT_FILES.read().is_empty() {
                if OUTPUT_FILES.read()[index].ends_with("svg") {
                    img {
                        style: "padding-top: 30px; padding-bottom: 10px; align-self: center;",
                        fill: "#fff",
                        width: "200",
                        height: "200",
                        src: OUTPUT_FILES.read()[index].clone(),
                    }
                    p { style: "font-size: 24px;",
                        if let Some(message) = STATUS_MESSAGE.read().clone() {
                            "{message}"
                        } else {
                            "Processing..."
                        }
                    }
                } else {
                    // Display the current image
                    div { style: "height: 100%; width: 100%; align-content: center",
                        img {
                            src: load_image_from_file(OUTPUT_FILES.read()[index].clone()),
                            alt: "Processed Image",
                            style: "max-width: 100%;
                            max-height: 100%;
                            object-fit: contain;",
                        }
                    }
                }
            } else {
                p { style: "font-size: 24px;", "No image available" }
            }
        }
    }
}

fn load_image_from_file(path: String) -> String {
    use base64::prelude::*;

    let file_content = std::fs::read(path).expect("Failed to read image file");

    let encoded_image = BASE64_STANDARD.encode(file_content);

    format!("data:image/png;base64,{}", encoded_image)
}

/// Runs the model using the provided parameters.
/// This function spawns a new process using the `nix run` command.
///
/// # Arguments
/// * `username` - The Earthdata username.
/// * `password` - The Earthdata password.
/// * `wkt_string` - The WKT string for the area of interest.
/// * `date` - The date string in the format "YYYY-MM-DDTHH:MM:SS.mmm".
///
/// # Returns
/// This function does not return a value. It spawns a process and waits for it to finish.
/// The output file path is stored in the `OUTPUT_FILE` global signal.
async fn run_model(username: &str, password: &str, wkt_string: &str, date: &str) {
    let mut child = Command::new("nix")
        .arg("run")
        .arg(".")
        .arg("--")
        .arg(username)
        .arg(password)
        .arg(format!("\"{wkt_string}\""))
        .arg(date)
        .spawn()
        .expect("Failed to spawn child process");

    let mut success = false;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                success = status.success();
                break;
            }
            Ok(None) => {
                // Check the log file for a new status message
                let status_log_path = format!("assets/tmp/{}/status.txt", date);
                let status_message = std::fs::read_to_string(&status_log_path)
                    .unwrap_or_else(|_| "Processing...".to_string());

                *STATUS_MESSAGE.write() = Some(status_message);

                // Process is still running, wait a bit before checking again
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            Err(e) => {
                eprintln!("Error waiting for child process: {}", e);
                break;
            }
        }
    }

    // Reset status message
    *STATUS_MESSAGE.write() = None;

    // Drop trhobber
    std::mem::drop(OUTPUT_FILES.write().pop());

    // Look for file in assets/tmp/{date}/output.png
    if success {
        OUTPUT_FILES
            .write()
            .push(format!("assets/tmp/{}/output.png", date));
    }
}
