// Dylan Maltos
// EmberCastAI - webScraper.go
// Scrapes KMZ files from ftp.wildfire.gov into /kmz_data directory for extractPolygon.py to use
// Last modified: 2/12/25

package main

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/gocolly/colly"
)

// Function fileExists - Checks if a file already exists
func fileExists(filePath string) bool {
	_, err := os.Stat(filePath)
	return !os.IsNotExist(err)
}

// Function isValidKMZ - Checks if a KMZ file is valid and contains "Heat Perimeter"
func isValidKMZ(kmzPath string) bool {
	file, err := os.Open(kmzPath)
	if err != nil {
		fmt.Printf("Error opening KMZ file: %v\n", err)
		return false
	}
	defer file.Close()

	// Read file into a byte buffer
	data, err := io.ReadAll(file)
	if err != nil {
		fmt.Printf("Error reading KMZ file: %v\n", err)
		return false
	}

	// Check if it's a valid ZIP archive (KMZ is a ZIP)
	zipReader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		fmt.Printf("Invalid KMZ file: %s\n", kmzPath)
		return false
	}

	// Check for "Heat Perimeter" in the extracted KML contents
	for _, file := range zipReader.File {
		if strings.HasSuffix(file.Name, ".kml") {
			kmlFile, err := file.Open()
			if err != nil {
				continue
			}
			defer kmlFile.Close()

			kmlData, err := io.ReadAll(kmlFile)
			if err != nil {
				continue
			}

			if strings.Contains(string(kmlData), "Heat Perimeter") {
				return true
			}
		}
	}

	fmt.Printf("Skipping %s - No 'Heat Perimeter' layer found\n", kmzPath)
	return false
}

// Function downloadKMZ - Downloads KMZ file and saves it **only if valid**
func downloadKMZ(urlStr, savePath string) error {
	// Skip downloading if the file already exists
	if fileExists(savePath) {
		fmt.Printf("Skipping (already exists): %s\n", savePath)
		return nil
	}

	resp, err := http.Get(urlStr)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: %s", resp.Status)
	}

	// Read response body into memory
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	// Validate the KMZ file from memory before saving
	if !isValidKMZFromBytes(data) {
		return fmt.Errorf("invalid KMZ file: %s", urlStr)
	}

	// Ensure directory is created only when a valid KMZ file is confirmed
	os.MkdirAll(filepath.Dir(savePath), os.ModePerm)

	// Create and write the file
	out, err := os.Create(savePath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = out.Write(data)
	if err != nil {
		return err
	}

	fmt.Printf("Saved: %s\n", savePath)
	return nil
}

// Function isValidKMZFromBytes - Checks KMZ validity before writing to disk
func isValidKMZFromBytes(data []byte) bool {
	zipReader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return false
	}

	// Check for "Heat Perimeter" layer in KML files
	for _, file := range zipReader.File {
		if strings.HasSuffix(file.Name, ".kml") {
			kmlFile, err := file.Open()
			if err != nil {
				continue
			}
			defer kmlFile.Close()

			kmlData, err := io.ReadAll(kmlFile)
			if err != nil {
				continue
			}

			if strings.Contains(string(kmlData), "Heat Perimeter") {
				return true
			}
		}
	}
	return false
}

// Function getRelativePath - Extracts only the necessary directory structure from URL
func getRelativePath(baseURL, fullURL string) string {
	parsedBase, err := url.Parse(baseURL)
	if err != nil {
		return ""
	}
	parsedFull, err := url.Parse(fullURL)
	if err != nil {
		return ""
	}

	// Get relative path by removing the baseURL from fullURL
	relativePath := strings.TrimPrefix(parsedFull.Path, parsedBase.Path)

	// Remove leading slash if present
	relativePath = strings.TrimPrefix(relativePath, "/")

	return relativePath
}

// Function main - Scrapes KMZ files from ftp.wildfire.gov
func main() {
	// Root URL of ftp.wildfire.gov to scrape
	baseURL := "https://ftp.wildfire.gov/public/incident_specific_data/"

	// Save KMZ files in /kmz_data directory
	outputDir := "kmz_data"

	// Create a colly collector
	c := colly.NewCollector(
		colly.AllowedDomains("ftp.wildfire.gov"),
		colly.MaxDepth(0), // No depth limit
	)

	// Manually filter out unwanted URLs
	c.OnRequest(func(r *colly.Request) {
		if !strings.HasPrefix(r.URL.String(), baseURL) {
			fmt.Printf("Skipping non-target URL: %s\n", r.URL.String())
			r.Abort()
		}
	})

	// Callback for each discovered link
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")

		// Skip parent directory links and non-directory targets
		if link == "../" || link == "/" {
			return
		}

		fullURL := e.Request.AbsoluteURL(link)

		// Ensure we only visit URLs inside incident_specific_data/
		if !strings.HasPrefix(fullURL, baseURL) {
			return
		}

		// If it's a directory, visit it
		if strings.HasSuffix(link, "/") {
			c.Visit(fullURL)
			// If it's a KMZ file, download it
		} else if strings.HasSuffix(link, ".kmz") {
			relativePath := getRelativePath(baseURL, fullURL)
			savePath := filepath.Join(outputDir, relativePath)

			// Download and validate KMZ before creating directories
			err := downloadKMZ(fullURL, savePath)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			}
		}
	})

	// Handle errors during scraping
	c.OnError(func(_ *colly.Response, err error) {
		fmt.Printf("Error: %v\n", err)
	})

	// Start scraping
	fmt.Println("Starting scraper...")
	c.Visit(baseURL)
	fmt.Println("Scraping complete!")
}
