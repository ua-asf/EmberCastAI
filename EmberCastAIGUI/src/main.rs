#![allow(non_snake_case)]

use dioxus::{
    document::{Link, Style, Title},
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
            }}"#
        }
        div { style: "text-align: center;
            height: 100%;
            display: grid;
            gap: 20px;
            grid-template-columns: auto 5px 1fr;
            height: 100%;
            flex: 1,
            margin: 0px 0px;
            height: 100vh;",
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
        div { style: "padding: 20px;",
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
                    style: format!("border-color: {}", if password_error() { "red" } else { "white" }),
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
                        if !USERNAME.read().clone().is_some_and(|v| !v.is_empty()) {
                            username_error.set(true);
                            errors = true;
                        }
                        if !PASSWORD.read().clone().is_some_and(|v| !v.is_empty()) {
                            password_error.set(true);
                            errors = true;
                        }
                        if !WKT_STRING.read().clone().is_some_and(|v| !v.is_empty()) {
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
                        OUTPUT_FILES.write().push(THROBBER.to_string());
                        spawn(async move {
                            run_model(
                                &USERNAME().unwrap_or_default(),
                                &PASSWORD().unwrap_or_default(),
                                &WKT_STRING().unwrap_or_default(),
                                &formatted_date,
                            ).await;
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
        div { style: "width: 100%; height: 100%; background-color: #FFF;" }
    }
}

#[component]
fn RenderImage() -> Element {
    let output_files = OUTPUT_FILES();
    let files_count = output_files.len();

    let mut index: Signal<usize> = use_signal(|| 0);

    rsx! {
        div { style: "padding: 20px;",
            // Image navigation buttons
            div { style: "display: flex; justify-content: center; align-items: center; gap: 10px;",
                // Previous/Decrement button
                button {
                    disabled: index() == 0,
                    onclick: move |_| {
                        if index() > 0 {
                            index.set(index() - 1);
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
                    disabled: index() + 1 >= files_count,
                    onclick: move |_| {
                        if index() < files_count - 1 {
                            index.set(index() + 1);
                        }
                    },
                    "Next →"
                }
            }

            // Render the selected image if any are available
            if !output_files.is_empty() {
                if output_files[index()].ends_with("svg") {
                    img { style: "padding-top: 30px; padding-bottom: 10px;",
                        fill: "#fff",
                        width: "200",
                        height: "200",
                        src: output_files[index()].clone(),
                    }
                    p { style: "font-size: 24px;", "Processing..." }
                } else {
                    // Display the current image
                    img {
                        src: load_image_from_file(output_files[index()].clone()),
                        alt: "Processed Image",
                        style: "width: 100%; height: auto; object-fit: contain; padding: 10px; border: 1px solid #000;",
                    }
                }
            } else {
                p { style: "font-size: 24px;", "No image available" }
            }
        }
    }
}

fn load_image_from_file(path: String) -> String {
    let file_content = std::fs::read(path).expect("Failed to read image file");

    let encoded_image = base64::encode(file_content);

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
    let output = Command::new("nix")
        .arg("run")
        .arg(".")
        .arg("--")
        .arg(username)
        .arg(password)
        .arg(format!("\"{wkt_string}\""))
        .arg(date)
        .output()
        .await
        .expect("Failed to start child process");

    // Write output to log file
    let file_path = format!("assets/tmp/{}/log.txt", date);
    std::fs::create_dir_all(format!("assets/tmp/{}", date)).expect("Failed to create directory");
    std::fs::write(&file_path, std::str::from_utf8(&*output.stdout).unwrap())
        .expect("Failed to write log file");
    std::fs::write(
        format!("assets/tmp/{}/error.txt", date),
        std::str::from_utf8(&*output.stderr).unwrap(),
    )
    .expect("Failed to write error file");

    // Drop trhobber
    std::mem::drop(OUTPUT_FILES.write().pop());

    // Look for file in assets/tmp/{date}/output.png
    OUTPUT_FILES
        .write()
        .push(format!("assets/tmp/{}/output.png", date));
}
