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

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, UIGestureRecognizerDelegate {

    @IBOutlet weak var previewImageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setupEventImageView()
    }

    func predictUsingPixelBuffer(image: UIImage?) {
        let model = Inceptionv3()
        
        if let pixelBuffer = image?.pixelBuffer(width: 299, height: 299),
            let prediction = try? model.prediction(image: pixelBuffer) {
//            print(prediction.classLabel)
//            print(prediction.classLabelProbs)
            self.resultLabel.text = prediction.classLabel + "\n\(prediction.classLabelProbs[prediction.classLabel]!*100)%"
            previewImageView.image = image
        }
    }
    
    func predictWithVGG16UsingVision(image: UIImage) {
        let modelFile = VGG16()
        
        let model = try! VNCoreMLModel(for: modelFile.model)
        
        prepareModelToAnalyze(model: model, image: image)
    }
    
    func predictWithResnet50UsingVision(image: UIImage) {
        let modelFile = Resnet50()
        
        let model = try! VNCoreMLModel(for: modelFile.model)
        
        prepareModelToAnalyze(model: model, image: image)
    }
    
    func prepareModelToAnalyze(model: VNCoreMLModel, image: UIImage) {
        let ciImage = CIImage(image: image)
        let cgImage = convertCIImageToCGImage(inputImage: ciImage!)
        let handler = VNImageRequestHandler(cgImage: cgImage!, options: [:])
        let request = VNCoreMLRequest(model: model, completionHandler: resultsMethod)
        
        try! handler.perform([request])
    }
    
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage! {
        let context = CIContext(options: nil)
        
        return context.createCGImage(inputImage, from: inputImage.extent)
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

        self.resultLabel.text = bestPrediction + "\n\(bestConfidence*100)%"
    }
    
    @IBAction func resnet50ButtonTapped(_ sender: Any) {
        guard let image = previewImageView.image else {
            
            let alert = UIAlertController(title: "image required", message: nil, preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "Ok", style: UIAlertActionStyle.default, handler: nil))
            self.present(alert, animated: true, completion: nil)
            
            return }
        
        predictWithResnet50UsingVision(image: image)
    }
    
    @IBAction func vGG16ButtonTapped(_ sender: Any) {
        guard let image = previewImageView.image else {
            
            let alert = UIAlertController(title: "image required", message: nil, preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "Ok", style: UIAlertActionStyle.default, handler: nil))
            self.present(alert, animated: true, completion: nil)
            
            return }
        
        predictWithVGG16UsingVision(image: image)
    }
    
    @IBAction func inceptionV3ButtonTapped(_ sender: Any) {
        guard let image = previewImageView.image else {
            
            let alert = UIAlertController(title: "image required", message: nil, preferredStyle: UIAlertControllerStyle.alert)
            alert.addAction(UIAlertAction(title: "Ok", style: UIAlertActionStyle.default, handler: nil))
            self.present(alert, animated: true, completion: nil)
            
            return }
        
        self.predictUsingPixelBuffer(image: image)
    }
    private func setupEventImageView() {
        
        guard let imageView = previewImageView else { return }
        
        let tapRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTapEventImageView(sender: )))
        
        tapRecognizer.delegate = self
        
        imageView.addGestureRecognizer(tapRecognizer)
        
        imageView.isUserInteractionEnabled = true
    }
    
    @objc func handleTapEventImageView(sender: UITapGestureRecognizer) {
        
        let photoAlert = UIAlertController(title: "Pick An Image", message: nil, preferredStyle: .actionSheet)

        photoAlert.addAction(UIAlertAction(title: "Camera", style: .default, handler: { _ in
            
            self.openCamera()
            
        }))
        
        photoAlert.addAction(UIAlertAction(title: "Album", style: .default, handler: { _ in
            
            self.openAlbum()
            
        }))
        
        photoAlert.addAction(UIAlertAction(title: "Cancel", style: .default, handler: nil))
        
        self.present(photoAlert, animated: true)
        
    }
    
    func openCamera() {
        
        let isCameraExist = UIImagePickerController.isSourceTypeAvailable(.camera)
        
        if isCameraExist {
            
            imagePicker.delegate = self
            
            imagePicker.sourceType = .camera
            
            self.present(imagePicker, animated: true)
            
        } else {
            
            let noCaremaAlert = UIAlertController(title: "Sorry", message: "You don't have camera lol", preferredStyle: .alert)
            
            noCaremaAlert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
            
            self.present(noCaremaAlert, animated: true)
        }
    }
    
    func openAlbum() {
        
        imagePicker.delegate = self
        
        imagePicker.sourceType = .photoLibrary
        
        self.present(imagePicker, animated: true)
        
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        self.dismiss(animated: true) { () -> Void in
            
            if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
                
                self.predictUsingPixelBuffer(image: image)

                self.previewImageView.image = image
                
            } else {
                
                print("picked photo error.")
            }
        }
    }
}
