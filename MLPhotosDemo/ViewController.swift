//
//  ViewController.swift
//  MLPhotosDemo
//
//  Created by Kai-Ping Tseng on 2017/12/5.
//  Copyright © 2017年 Kai-Ping Tseng. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {

    @IBOutlet weak var resultLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        let path = Bundle.main.path(forResource: "dog", ofType: "jpg")
        guard let pathUnwrapped = path else { return }
            
        let imageURL = NSURL.fileURL(withPath: pathUnwrapped)
        
//        let modelFile = GoogLeNetPlaces()
        let modelFile = Resnet50()

        let model = try! VNCoreMLModel(for: modelFile.model)
        let handler = VNImageRequestHandler(url: imageURL)
        let request = VNCoreMLRequest(model: model, completionHandler: resultsMethod)
        
        try! handler.perform([request])
        
    }

    
    func resultsMethod(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNClassificationObservation] else {
            fatalError("Could not get results from ML Vision request.")
        }

        var bestPrediction = ""
        var bestConfidence: Float = 0.0
        
        for classification in results {
            if classification.confidence > bestConfidence {
                bestConfidence = classification.confidence
                bestPrediction = classification.identifier
            }
        }
        
        print("Predicted: \(bestPrediction) with confidence of \(bestConfidence) out of 1.")
        
        self.resultLabel.text = bestPrediction
    }
}

