#![allow(non_snake_case)]

use dioxus::{
    document::{Style, Title},
    prelude::*,
};

use serde_json::json;

fn main() {
    dioxus::launch(App);
}

pub static USERNAME: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static PASSWORD: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static WKT_STRING: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static OUTPUT_DATA: GlobalSignal<ProcessingState> =
    GlobalSignal::new(|| ProcessingState::Empty);
pub static STATUS_MESSAGE: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static INDEX: GlobalSignal<usize> = GlobalSignal::new(|| 0);

pub static THROBBER: Asset = asset!("assets/throbber.svg");

pub static API_ENDPOINT: &str = "http://127.0.0.1:8000";

struct IntImage(Vec<u8>);

enum ProcessingState {
    Empty,
    Processing(String), // Throbber or placeholder image location
    Processed {
        before: IntImage,
        after: IntImage,
        dem: IntImage,
    },
}

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
                text-align: center;
            }}"#
        }
        div { style: "text-align: center;
            height: 100%;
            width: 100vw;
            display: flex;
            flex-direction: column;
            gap: 20px;
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
        div { style: "display: flex; flex-direction: row; flex: 1 0 auto; justify-content: center; padding: 5px; gap: 20px",
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
                        *OUTPUT_DATA.write() = ProcessingState::Processing(THROBBER.to_string());
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
        div { style: "width: 100%; height: 5px; background-color: #FFF;" }
    }
}

#[component]
fn RenderImage() -> Element {
    rsx! {
        div { style: "display: flex; flex-direction: column; justify-content: start; align-items: center; height: 100vh; width: 100%; overflow: hidden",
            // Render the selected image if any are available
            match *OUTPUT_DATA.read() {
                ProcessingState::Empty => {
                    rsx! { p { style: "font-size: 24px;", "No image available" } }
                },
                ProcessingState::Processing(ref img_path) => {
                    rsx! {
                        img {
                            style: "padding-top: 30px; padding-bottom: 10px; align-self: center;",
                            fill: "#fff",
                            width: "200",
                            height: "200",
                            src: "{img_path}",
                        }
                        p { style: "font-size: 24px;",
                            if let Some(message) = STATUS_MESSAGE.read().clone() {
                                "{message}"
                            } else {
                                "Processing..."
                            }
                        }
                    }
                },
                ProcessingState::Processed { ref before, ref after, ref dem } => {
                    rsx! {
                        div { style: "display: flex; flex-direction: row; gap: 20px; justify-content: center; align-items: center; padding-top: 30px; padding-bottom: 10px;",
                            div { style: "display: flex; flex-direction: column; gap: 10px; justify-content: center; align-items: center;",
                                p { "Before" }
                                BrightnessImage { brightness_data: before.0.clone(), color: (255, 0, 0), dem: dem.0.clone() }
                            }
                            div { style: "display: flex; flex-direction: column; gap: 10px; justify-content: center; align-items: center;",
                                p { "After" }
                                BrightnessImage { brightness_data: after.0.clone(), color: (255, 0, 0), dem: dem.0.clone() }
                            }
                        }
                    }
                }
            }
        }
    }
}

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{ImageBuffer, Luma};

/// Converts grayscale brightness values to a PNG data URL
fn brightness_to_data_url(brightness: &[u8], width: u32, height: u32) -> Option<String> {
    // Create grayscale image from brightness values
    let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, brightness)?;

    // Encode to PNG in memory
    let mut png_bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .ok()?;

    Some(format!(
        "data:image/png;base64,{}",
        STANDARD.encode(&png_bytes)
    ))
}

#[component]
fn BrightnessImage(brightness_data: Vec<u8>, color: (u8, u8, u8), dem: Vec<u8>) -> Element {
    let width = brightness_data.len().isqrt() as u32;
    let height = width;

    let data_url = use_memo(move || {
        brightness_to_data_url(&brightness_data, width, height).unwrap_or_default()
    });

    rsx! {
        if !data_url().is_empty() {
            img {
                src: "{data_url}",
                alt: "Brightness map",
                width: "{width}",
                height: "{height}"
            }
        } else {
            div { "Invalid image dimensions" }
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
    // Strip all of the possible prefix combinations from the WKT string
    let wkt_prefix_stripped = strip_prefixes(wkt_string, &["POLYGON", " ", "((", " "]);

    let wkt_stripped = strip_suffixes(wkt_prefix_stripped, &[" ", "))"]);

    println!("Stripped WKT: {}", wkt_stripped);

    // Parse coordinates into actual numbers, not strings
    let wkt_points: Vec<Vec<[f64; 2]>> = vec![
        wkt_stripped
            .split(", ")
            .map(|v| {
                let (x, y) = v.split_once(" ").unwrap();
                [x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap()]
            })
            .collect(),
    ];

    // Use serde_json for proper structure
    let data = json!({
        "username": username,
        "password": password,
        "wkt_points": wkt_points,
        "date_str": date,
    });

    let client = reqwest::Client::new();

    let response = match client
        .post(format!("{}/process_wkt", API_ENDPOINT))
        .json(&data)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            *STATUS_MESSAGE.write() = Some(format!("Error: Failed to send request - {}", e));
            return;
        }
    };

    if response.status() != 200 {
        *STATUS_MESSAGE.write() = Some(format!(
            "Error: Received status code {}, {}",
            response.status(),
            response.text().await.unwrap_or_default()
        ));
        return;
    }

    // Reset status message
    *STATUS_MESSAGE.write() = None;

    let result = response.text().await.unwrap_or_default();

    // Process result into two IntImage structs
    // Turn response into json

    let json = serde_json::from_str::<serde_json::Value>(&result).unwrap_or_default();

    // Get "original" Vec<u8>

    let original = json["original"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect::<Vec<u8>>();

    let results = json["results"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect::<Vec<u8>>();

    let dem = json["dem"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect::<Vec<u8>>();

    *OUTPUT_DATA.write() = ProcessingState::Processed {
        before: IntImage(original),
        after: IntImage(results),
        dem: IntImage(dem),
    };
}

fn strip_prefixes<'a>(string: &'a str, prefixes: &'a [&'a str]) -> &'a str {
    let mut result = string;
    for prefix in prefixes {
        if result.starts_with(prefix) {
            result = &result[prefix.len()..];
        }
    }
    result
}

fn strip_suffixes<'a>(string: &'a str, suffixes: &'a [&'a str]) -> &'a str {
    let mut result = string;
    for suffix in suffixes {
        if result.ends_with(suffix) {
            result = &result[..result.len() - suffix.len()];
        }
    }
    result
}

fn truncate_string_float(s: &str, decimal_places: usize) -> String {
    if let Some(dot_index) = s.find('.') {
        // Find the index where truncation should occur
        let end_index = dot_index + 1 + decimal_places;

        // Ensure end_index doesn't exceed the string length
        if end_index >= s.len() {
            s.to_string() // Return the original string if not enough decimal places
        } else {
            s[..end_index].to_string() // Slice and return the truncated string
        }
    } else {
        s.to_string() // No decimal point found, return the original string
    }
}
