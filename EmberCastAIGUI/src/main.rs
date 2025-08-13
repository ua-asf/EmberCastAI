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

pub static THROBBER: Asset = asset!("assets/throbber.gif");

#[component]
pub fn App() -> Element {
    rsx! {
        Title { "EmberCast AI" }
        // Stylesheet
        // Black background with white text
        Style {
            "body {{
                background-color: #020202;
                color: #FEFEFE;
                margin: 0;
             }}"
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
    let mut button_clickable = use_signal(|| true);

    rsx! {
        div { style: "padding: 20px;",
            div {
                p { "Earthdata Username:" }
                input {
                    oninput: move |e| {
                        *USERNAME.write() = Some(e.value().clone());
                    },
                    value: USERNAME(),
                }
            }

            div {
                p { "Earthdata Password:" }
                input {
                    r#type: "password",
                    oninput: move |e| {
                        *PASSWORD.write() = Some(e.value().clone());
                    },
                    value: PASSWORD(),
                }
            }

            div {
                p { "WKT String:" }
                input {
                    oninput: move |e| {
                        *WKT_STRING.write() = Some(e.value().clone());
                    },
                    value: WKT_STRING(),
                }
            }

            div { style: "margin-top: 20px; display: flex; justify-content: center;",
                button {
                    style: "width: 100%; max-width: 200px; padding: 5px; font-size: 16px;",
                    onclick: move |_| {
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
            div { style: "display: flex; justify-content: center; gap: 10px;",
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
                // Display the current image
                img {
                    src: output_files[index()].clone(),
                    alt: "Processed Image",
                    style: "width: 100%; height: auto; object-fit: contain; padding: 10px; border: 1px solid #000;",
                }
            } else {
                p { "No image available." }
            }
        }
    }
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
    let status = Command::new("nix")
        .arg("run")
        .arg(".")
        .arg("--")
        .arg(username)
        .arg(password)
        .arg(format!("\"{wkt_string}\""))
        .arg(date)
        .status()
        .await
        .expect("Failed to start child process");

    println!("Nix run finished with status: {}", status);

    if !status.success() {
        // Remove the last output file placeholder
        std::mem::drop(OUTPUT_FILES.write().pop());
        return;
    }

    // Look for file in assets/tmp/{date}/output.png
    OUTPUT_FILES.write()[OUTPUT_FILES.len() - 1] = format!("assets/tmp/{}/output.png", date);
}
