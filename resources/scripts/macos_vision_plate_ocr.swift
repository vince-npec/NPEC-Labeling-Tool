import CoreGraphics
import Foundation
import ImageIO
import Vision

struct OCRCandidate: Codable {
    let text: String
    let confidence: Float
}

struct OCRObservation: Codable {
    let candidates: [OCRCandidate]
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct BarcodeObservation: Codable {
    let payload: String
    let symbology: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct OCRResult: Codable {
    let path: String
    let observations: [OCRObservation]
    let barcodes: [BarcodeObservation]
    let error: String?
}

func recognize(path: String) -> OCRResult {
    let url = URL(fileURLWithPath: path) as CFURL
    guard
        let source = CGImageSourceCreateWithURL(url, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        return OCRResult(path: path, observations: [], barcodes: [], error: "Unable to decode image")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.minimumTextHeight = 0.025
    let barcodeRequest = VNDetectBarcodesRequest()

    do {
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request, barcodeRequest])
        let observations = (request.results ?? []).map { observation in
            OCRObservation(
                candidates: observation.topCandidates(4).map {
                    OCRCandidate(text: $0.string, confidence: $0.confidence)
                },
                x: observation.boundingBox.origin.x,
                y: observation.boundingBox.origin.y,
                width: observation.boundingBox.width,
                height: observation.boundingBox.height
            )
        }.sorted {
            if abs($0.y - $1.y) > 0.03 {
                return $0.y > $1.y
            }
            return $0.x < $1.x
        }
        let barcodes = (barcodeRequest.results ?? []).compactMap { observation -> BarcodeObservation? in
            guard let payload = observation.payloadStringValue, !payload.isEmpty else {
                return nil
            }
            return BarcodeObservation(
                payload: payload,
                symbology: observation.symbology.rawValue,
                confidence: observation.confidence,
                x: observation.boundingBox.origin.x,
                y: observation.boundingBox.origin.y,
                width: observation.boundingBox.width,
                height: observation.boundingBox.height
            )
        }
        return OCRResult(path: path, observations: observations, barcodes: barcodes, error: nil)
    } catch {
        return OCRResult(path: path, observations: [], barcodes: [], error: String(describing: error))
    }
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
for path in CommandLine.arguments.dropFirst() {
    autoreleasepool {
        let result = recognize(path: path)
        if let data = try? encoder.encode(result), let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }
}
